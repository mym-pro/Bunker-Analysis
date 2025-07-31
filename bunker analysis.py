import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import logging
from io import BytesIO
from datetime import datetime
from github import Github
import base64
import re
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 常量定义 - 只保留核心数据
EXTRACTION_CONFIG = {
    1: [
        r"(MFSPD00)\s+(\d+\.\d+)",  # Singapore
        r"(MFRDD00)\s+(\d+\.\d+)",  # Rotterdam
        r"(MFHKD00)\s+(\d+\.\d+)",  # Hong Kong
        r"(MFSAD00)\s+(\d+\.\d+)",  # Santos
        r"(MFZSD00)\s+(\d+\.\d+)"   # Zhoushan
    ],
    2: [
        r"(MLBSO00)\s+(\d+\.\d+)",  # Low Sulphur Fuel Oil
        r"(LNBSF00)\s+(\d+\.\d+)"   # LNG Bunker
    ]
}

DATE_PATTERN = r"Volume\s+\d+\s+/\s+Issue\s+\d+\s+/\s+(\w+\s+\d{1,2},\s+\d{4})"

# 只保留核心港口和燃料数据
CORE_PORTS = ["MFSPD00", "MFRDD00", "MFHKD00", "MFSAD00", "MFZSD00"]
FUEL_TYPES = ["MLBSO00", "LNBSF00"]

# GitHub 数据管理类
class GitHubDataManager:
    def __init__(self, token: str, repo_name: str):
        self.token = token
        self.repo_name = repo_name
        self.g = Github(self.token)
        self.repo = self.g.get_repo(self.repo_name)
    
    @st.cache_data(ttl=3600)
    def read_excel(_self, file_path: str) -> pd.DataFrame:
        try:
            contents = _self.repo.get_contents(file_path)
            return pd.read_excel(BytesIO(base64.b64decode(contents.content)), True)
        except:
            return pd.DataFrame(), False
    
    def save_excel(_self, df: pd.DataFrame, file_path: str, commit_msg: str) -> bool:
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            content = output.getvalue()
            
            st.cache_data.clear()
            
            try:
                contents = _self.repo.get_contents(file_path)
                _self.repo.update_file(contents.path, commit_msg, base64.b64encode(content).decode(), contents.sha)
            except:
                _self.repo.create_file(file_path, commit_msg, base64.b64encode(content).decode())
            return True
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
            return False

# 高效PDF处理器
class FastBunkerPriceExtractor:
    def __init__(self, pdf_bytes: bytes, bunker_path: str, fuel_path: str):
        self.pdf_bytes = pdf_bytes
        self.bunker_path = bunker_path
        self.fuel_path = fuel_path
        self.gh_manager = GitHubDataManager(
            st.secrets.github.token, 
            st.secrets.github.repo
        )

    def process_pdf(self) -> bool:
        try:
            with fitz.open(stream=self.pdf_bytes, filetype="pdf") as doc:
                # 提取港口数据
                if doc.page_count > 0:
                    bunker_df = self._process_page(doc[0], is_bunker=True)
                    if bunker_df is not None:
                        self._save_data(bunker_df, self.bunker_path)
                
                # 提取燃料数据
                if doc.page_count > 1:
                    fuel_df = self._process_page(doc[1], is_bunker=False)
                    if fuel_df is not None:
                        self._save_data(fuel_df, self.fuel_path)
                
                return True
        except:
            return False

    def _process_page(self, page, is_bunker: bool) -> Optional[pd.DataFrame]:
        text = page.get_text().strip()
        date_match = re.search(DATE_PATTERN, text)
        if not date_match:
            return None
        
        try:
            date = datetime.strptime(date_match.group(1), "%B %d, %Y").date()
        except:
            return None

        # 只提取核心数据
        data = {'Date': [date]}
        config = EXTRACTION_CONFIG[1] if is_bunker else EXTRACTION_CONFIG[2]
        
        for pattern in config:
            match = re.search(pattern, text)
            if match:
                code, value = match.groups()
                data[code] = [float(value)]
        
        return pd.DataFrame(data)

    def _save_data(self, new_df: pd.DataFrame, output_path: str) -> bool:
        try:
            existing_df, exists = self.gh_manager.read_excel(output_path)
            if exists and not existing_df.empty:
                # 移除同日期旧数据
                existing_df = existing_df[existing_df['Date'] != new_df['Date'].iloc[0]]
                combined_df = pd.concat([existing_df, new_df])
            else:
                combined_df = new_df
            
            # 按日期排序
            combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.date
            combined_df = combined_df.sort_values('Date', ascending=False)
            
            return self.gh_manager.save_excel(
                combined_df,
                output_path,
                f"Update at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            )
        except:
            return False

# 主界面
def main_ui():
    st.set_page_config(page_title="船燃价格分析", layout="centered")
    st.title("船燃价格分析系统")
    
    # 文件上传区域
    uploaded_file = st.file_uploader("上传Bunkerwire PDF报告", type="pdf")
    if uploaded_file and st.button("处理文件"):
        with st.spinner("处理中..."):
            extractor = FastBunkerPriceExtractor(
                uploaded_file.getvalue(),
                "data/bunker_prices.xlsx",
                "data/fuel_prices.xlsx"
            )
            if extractor.process_pdf():
                st.success("文件处理成功！")
                st.cache_data.clear()
            else:
                st.error("文件处理失败，请检查格式")
    
    # 加载数据
    bunker_df = load_data("data/bunker_prices.xlsx", CORE_PORTS)
    fuel_df = load_data("data/fuel_prices.xlsx", FUEL_TYPES)
    
    st.divider()
    
    # 港口价格对比
    st.subheader("港口价格对比")
    if not bunker_df.empty:
        # 日期选择
        dates = bunker_df['Date'].unique()
        date1, date2 = st.select_slider(
            "选择对比日期",
            options=dates,
            value=(dates[0], dates[-1] if len(dates) > 1 else dates[0]))
        
        # 获取数据
        df1 = bunker_df[bunker_df['Date'] == date1]
        df2 = bunker_df[bunker_df['Date'] == date2]
        
        # 创建对比表格
        comparison = []
        for port in CORE_PORTS:
            price1 = df1[port].values[0] if port in df1.columns else None
            price2 = df2[port].values[0] if port in df2.columns else None
            
            if price1 and price2:
                change = (price1 - price2) / price2 * 100
                comparison.append({
                    "港口": port,
                    date1: f"${price1:.2f}",
                    date2: f"${price2:.2f}",
                    "变化": f"{change:+.2f}%"
                })
        
        if comparison:
            st.dataframe(pd.DataFrame(comparison).set_index("港口"))
        else:
            st.warning("无足够数据对比")
    else:
        st.warning("暂无港口数据")
    
    st.divider()
    
    # 燃料价格展示
    st.subheader("替代燃料价格")
    if not fuel_df.empty:
        # 只显示最新5条记录
        st.dataframe(fuel_df.head(5).set_index("Date"))
    else:
        st.warning("暂无燃料数据")

@st.cache_data(ttl=600)
def load_data(path: str, columns: list) -> pd.DataFrame:
    try:
        gh_manager = GitHubDataManager(
            st.secrets.github.token, 
            st.secrets.github.repo
        )
        df, exists = gh_manager.read_excel(path)
        if exists:
            # 确保日期格式正确
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            # 只保留需要的列
            return df[['Date'] + [col for col in columns if col in df.columns]]
    except:
        pass
    return pd.DataFrame()

if __name__ == "__main__":
    main_ui()

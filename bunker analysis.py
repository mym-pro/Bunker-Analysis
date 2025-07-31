import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging
from github import Github
import base64
from io import BytesIO

# 配置日志系统
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 常量定义
FUEL_COLUMNS = ["Date", "MLBSO00", "LNBSF00"]
COMPARE_PORT_CODES = ["MFSPD00", "MFRDD00", "MFHKD00", "MFSAD00", "MFZSD00"]
PORT_CODE_MAPPING = {
    "MFSPD00": "Singapore", "MFRDD00": "Rotterdam", "MFHKD00": "Hong Kong",
    "MFSAD00": "Santos", "MFZSD00": "Zhoushan"
}

# 数据处理类
class BunkerDataProcessor:
    @staticmethod
    def format_date(date_series: pd.Series) -> pd.Series:
        return pd.to_datetime(date_series, errors='coerce').dt.date

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if 'Date' in df.columns and not df.empty:
            df['Date'] = BunkerDataProcessor.format_date(df['Date'])
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date', ascending=False)
            df = df.drop_duplicates(subset='Date', keep='first')
        return df.sort_values('Date', ascending=True).reset_index(drop=True)

    @staticmethod
    def merge_data(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        if existing_df.empty:
            return new_df
        
        # 检查日期是否已存在
        new_date = new_df['Date'].iloc[0]
        if new_date in existing_df['Date'].values:
            # 覆盖该日期的数据
            existing_df = existing_df[existing_df['Date'] != new_date]
        
        return pd.concat([existing_df, new_df], ignore_index=True)

# GitHub 数据管理类
class GitHubDataManager:
    def __init__(self, token: str, repo_name: str):
        self.token = token
        self.repo_name = repo_name
        self.g = Github(self.token)
        self.repo = self.g.get_repo(self.repo_name)

    @st.cache_data(ttl=3600, show_spinner="从GitHub加载数据...")
    def read_excel(self, file_path: str) -> pd.DataFrame:
        try:
            contents = self.repo.get_contents(file_path)
            return pd.read_excel(BytesIO(base64.b64decode(contents.content)), engine='openpyxl')
        except Exception as e:
            logger.warning(f"GitHub读取失败: {str(e)}")
            return pd.DataFrame(columns=FUEL_COLUMNS)

    def save_excel(self, df: pd.DataFrame, file_path: str, commit_msg: str) -> bool:
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            content = output.getvalue()
            
            try:
                contents = self.repo.get_contents(file_path)
                self.repo.update_file(contents.path, commit_msg, base64.b64encode(content).decode(), contents.sha)
            except Exception:
                self.repo.create_file(file_path, commit_msg, base64.b64encode(content).decode())
            return True
        except Exception as e:
            logger.error(f"GitHub保存失败: {str(e)}")
            return False

# Streamlit 界面
def main_ui():
    st.set_page_config(page_title="船燃价格分析系统", layout="wide")
    st.title("Mariners' Bunker Price Analysis System")

    FUEL_PATH = "data/fuel_prices_new.xlsx"
    BUNKER_PATH = "data/bunker_prices_new.xlsx"

    # 加载数据
    gh_manager = GitHubDataManager(st.secrets.github.token, st.secrets.github.repo)
    fuel_df = gh_manager.read_excel(FUEL_PATH)
    bunker_df = gh_manager.read_excel(BUNKER_PATH)

    # 燃料价格分析
    st.subheader("替代燃料价格趋势")
    
    if not fuel_df.empty:
        # 显示最新数据
        st.dataframe(
            fuel_df.sort_values('Date', ascending=False).head(10).set_index("Date"),
            use_container_width=True
        )
        
        # 创建趋势图
        fig = go.Figure()
        for fuel_type in FUEL_COLUMNS:
            if fuel_type != 'Date' and fuel_type in fuel_df.columns:
                fig.add_trace(go.Scatter(
                    x=fuel_df['Date'],
                    y=fuel_df[fuel_type],
                    name=fuel_type,
                    mode='lines+markers',
                    connectgaps=True
                ))
        
        fig.update_layout(
            title="替代燃料价格趋势",
            xaxis_title="日期",
            yaxis_title="价格 (USD)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("暂无燃料数据可供分析。")

    # 数据对比
    st.subheader("指定日期港口价格对比")
    
    if not bunker_df.empty:
        # 获取可用日期
        date_options = bunker_df['Date'].dt.strftime('%Y-%m-%d').unique().tolist()
        date_options.sort(reverse=True)
        
        col1, col2 = st.columns(2)
        with col1:
            date1 = st.selectbox("选择对比日期1", date_options)
        with col2:
            date2 = st.selectbox("选择对比日期2", date_options, index=1 if len(date_options) > 1 else 0)
        
        if date1 and date2:
            df1 = bunker_df[bunker_df['Date'].dt.strftime('%Y-%m-%d') == date1]
            df2 = bunker_df[bunker_df['Date'].dt.strftime('%Y-%m-%d') == date2]
            
            if not df1.empty and not df2.empty:
                comparison = []
                for port in COMPARE_PORT_CODES:
                    if port in df1.columns and port in df2.columns:
                        price1 = df1[port].values[0]
                        price2 = df2[port].values[0]
                        
                        if pd.notna(price1) and pd.notna(price2):
                            change = ((price1 - price2) / price2 * 100)
                            comparison.append({
                                "港口": f"{PORT_CODE_MAPPING.get(port, port)} ({port})",
                                date1: price1,
                                date2: price2,
                                "变化 (%)": f"{change:.2f}%"
                            })
                
                if comparison:
                    st.dataframe(
                        pd.DataFrame(comparison).set_index("港口"),
                        use_container_width=True
                    )
                else:
                    st.warning("未找到选定日期的数据或选定港口的数据不完整。")
            else:
                st.warning("未找到选定日期的数据。")
    else:
        st.warning("暂无油价数据可供对比。")

    # 文件上传区域
    with st.expander("📤 上传新数据", expanded=True):
        uploaded_files = st.file_uploader(
            "选择Bunkerwire PDF报告（支持多选）",
            type=["pdf"],
            accept_multiple_files=True
        )

    # 文件处理
    if uploaded_files:
        with st.status("正在解析文件...", expanded=True) as status:
            for file in uploaded_files:
                try:
                    # 模拟从PDF提取数据的逻辑
                    new_data = pd.DataFrame({
                        "Date": [datetime.now().date()],
                        "MLBSO00": [100.0],
                        "LNBSF00": [200.0]
                    })
                    
                    # 合并新数据
                    combined_df = BunkerDataProcessor.merge_data(fuel_df, new_data)
                    combined_df = BunkerDataProcessor.clean_dataframe(combined_df)
                    
                    # 保存到GitHub
                    if gh_manager.save_excel(
                        combined_df,
                        FUEL_PATH,
                        f"Update fuel prices at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    ):
                        st.toast(f"✅ {file.name} 处理成功")
                    else:
                        st.toast(f"⚠️ {file.name} 保存失败")
                except Exception as e:
                    st.toast(f"❌ {file.name} 处理失败: {str(e)}")
        
            status.update(label="处理完成！", state="complete")

# 主程序入口
if __name__ == "__main__":
    main_ui()

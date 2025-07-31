import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import logging
import tempfile
from io import BytesIO
from datetime import datetime
from github import Github
import base64
import plotly.graph_objects as go
import re
import os
from typing import Dict
from typing import Optional
from typing import List

# --------------------------
# 配置日志系统
# --------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# --------------------------
# 常量定义
# --------------------------
EXTRACTION_CONFIG = {
    1: [
        r"(MFSPD00)\s+(\d+\.\d+)", r"(MFFJD00)\s+(\d+\.\d+)", r"(MFJPD00)\s+(\d+\.\d+)", 
        r"(BAMFB00)\s+(\d+\.\d+)", r"(MFSKD00)\s+(\d+\.\d+)", r"(WKMFA00)\s+(\d+\.\d+)", 
        r"(MFHKD00)\s+(\d+\.\d+)", r"(MFSHD00)\s+(\d+\.\d+)", r"(MFZSD00)\s+(\d+\.\d+)", 
        r"(MFDSY00)\s+(\d+\.\d+)", r"(MFDMB00)\s+(\d+\.\d+)", r"(MFDKW00)\s+(\d+\.\d+)", 
        r"(MFDKF00)\s+(\d+\.\d+)", r"(MFDMM00)\s+(\d+\.\d+)", r"(MFDCL00)\s+(\d+\.\d+)", 
        r"(MFAGD00)\s+(\d+\.\d+)", r"(MFDBD00)\s+(\d+\.\d+)", r"(MFGBD00)\s+(\d+\.\d+)", 
        r"(MFMLD00)\s+(\d+\.\d+)", r"(MFPRD00)\s+(\d+\.\d+)", r"(MFRDD00)\s+(\d+\.\d+)", 
        r"(MFDAN00)\s+(\d+\.\d+)", r"(MFDGT00)\s+(\d+\.\d+)", r"(MFDHB00)\s+(\d+\.\d+)", 
        r"(MFDIS00)\s+(\d+\.\d+)", r"(MFDLP00)\s+(\d+\.\d+)", r"(MFDNV00)\s+(\d+\.\d+)", 
        r"(MFDPT00)\s+(\d+\.\d+)", r"(MFLIS00)\s+(\d+\.\d+)", r"(MFLOM00)\s+(\d+\.\d+)", 
        r"(MFHOD00)\s+(\d+\.\d+)", r"(MFNYD00)\s+(\d+\.\d+)", r"(MFLAD00)\s+(\d+\.\d+)", 
        r"(MFNOD00)\s+(\d+\.\d+)", r"(MFPAD00)\s+(\d+\.\d+)", r"(MFSED00)\s+(\d+\.\d+)", 
        r"(MFVAD00)\s+(\d+\.\d+)", r"(MFBAD00)\s+(\d+\.\d+)", r"(MFCRD00)\s+(\d+\.\d+)", 
        r"(MFSAD00)\s+(\d+\.\d+)", r"(AMFVA00)\s+(\d+\.\d+)", r"(AMFCA00)\s+(\d+\.\d+)", 
        r"(AMFGY00)\s+(\d+\.\d+)", r"(AMFLB00)\s+(\d+\.\d+)", r"(AMFMT00)\s+(\d+\.\d+)", 
        r"(AMFSF00)\s+(\d+\.\d+)", r"(AMFMO00)\s+(\d+\.\d+)"
    ],
    2: [
        r"(MLBSO00)\s+(\d+\.\d+)", r"(LNBSF00)\s+(\d+\.\d+)"
    ]
}

DATE_PATTERN = r"Volume\s+\d+\s+/\s+Issue\s+\d+\s+/\s+(\w+\s+\d{1,2},\s+\d{4})"

PORT_CODE_MAPPING = {
    # Asia Pacific/Middle East
    "MFSPD00": "Singapore", "MFFJD00": "Fujairah", "MFJPD00": "Japan", "BAMFB00": "West Japan",
    "MFSKD00": "South Korea", "WKMFA00": "South Korea (West)", "MFHKD00": "Hong Kong",
    "MFSHD00": "Shanghai", "MFZSD00": "Zhoushan", "MFDSY00": "Sydney", "MFDMB00": "Melbourne",
    "MFDKW00": "Kuwait", "MFDKF00": "Khor Fakkan", "MFDMM00": "Mumbai", "MFDCL00": "Colombo",
    # Europe
    "MFAGD00": "Algeciras", "MFDBD00": "Durban", "MFGBD00": "Gibraltar", "MFMLD00": "Malta",
    "MFPRD00": "Piraeus", "MFRDD00": "Rotterdam", "MFDAN00": "Antwerp", "MFDGT00": "Gothenburg",
    "MFDHB00": "Hamburg", "MFDIS00": "Istanbul", "MFDLP00": "Las Palmas", "MFDNV00": "Novorossiisk",
    "MFDPT00": "St. Petersburg", "MFLIS00": "Lisbon", "MFLOM00": "Lome",
    # Americas
    "MFHOD00": "Houston", "MFNYD00": "New York", "MFLAD00": "Los Angeles", "MFNOD00": "New Orleans",
    "MFPAD00": "Philadelphia", "MFSED00": "Seattle", "MFVAD00": "Vancouver", "MFBAD00": "Buenos Aires",
    "MFCRD00": "Cartagena", "MFSAD00": "Santos", "AMFVA00": "Valparaiso", "AMFCA00": "Callao",
    "AMFGY00": "Guayaquil", "AMFLB00": "La Libertad", "AMFMT00": "Montevideo", "AMFSF00": "San Francisco",
    "AMFMO00": "Montreal*"
}

BUNKER_COLUMNS = [
    "Date", "MFSPD00", "MFFJD00", "MFJPD00", "BAMFB00", "MFSKD00", "WKMFA00", "MFHKD00",
    "MFSHD00", "MFZSD00", "MFDSY00", "MFDMB00", "MFDKW00", "MFDKF00", "MFDMM00", "MFDCL00",
    "MFAGD00", "MFDBD00", "MFGBD00", "MFMLD00", "MFPRD00", "MFRDD00", "MFDAN00",
    "MFDGT00", "MFDHB00", "MFDIS00", "MFDLP00", "MFDNV00", "MFDPT00", "MFLIS00", "MFLOM00",
    "MFHOD00", "MFNYD00", "MFLAD00", "MFNOD00", "MFPAD00", "MFSED00", "MFVAD00",
    "MFBAD00", "MFCRD00", "MFSAD00", "AMFVA00", "AMFCA00", "AMFGY00", "AMFLB00",
    "AMFMT00", "AMFSF00", "AMFMO00"
]

FUEL_COLUMNS = ["Date", "MLBSO00", "LNBSF00"]

COMPARE_PORT_CODES = ["MFSPD00", "MFRDD00", "MFHKD00", "MFSAD00", "MFZSD00"]

# --------------------------
# 数据处理类
# --------------------------
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

# --------------------------
# GitHub 数据管理类
# --------------------------
class GitHubDataManager:
    def __init__(self, token: str, repo_name: str):
        self.token = token
        self.repo_name = repo_name
        self.g = Github(self.token)
        self.repo = self.g.get_repo(self.repo_name)
    
    @st.cache_data(ttl=3600, show_spinner="从GitHub加载数据...")
    def read_excel(_self, file_path: str) -> pd.DataFrame:
        try:
            contents = _self.repo.get_contents(file_path)
            return pd.read_excel(BytesIO(base64.b64decode(contents.content)), True)
        except Exception as e:
            logger.warning(f"GitHub读取失败: {str(e)}")
            return pd.DataFrame(), False
    
    def save_excel(_self, df: pd.DataFrame, file_path: str, commit_msg: str) -> bool:
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            content = output.getvalue()
            
            # 清除读取缓存
            st.cache_data.clear()
            
            try:
                contents = _self.repo.get_contents(file_path)
                _self.repo.update_file(contents.path, commit_msg, base64.b64encode(content).decode(), contents.sha)
            except:
                _self.repo.create_file(file_path, commit_msg, base64.b64encode(content).decode())
            return True
        except Exception as e:
            logger.error(f"GitHub保存失败: {str(e)}")
            return False

# --------------------------
# PDF 数据提取类
# --------------------------
class EnhancedBunkerPriceExtractor:
    def __init__(self, pdf_bytes: bytes, bunker_path: str, fuel_path: str):
        self.pdf_bytes = pdf_bytes
        self.bunker_path = bunker_path
        self.fuel_path = fuel_path
        self.gh_manager = GitHubDataManager(
            st.secrets.github.token, 
            st.secrets.github.repo
        )

    def process_pdf(self) -> Dict[str, int]:
        result = {'bunker': 0, 'fuel': 0}
        
        try:
            with fitz.open(stream=self.pdf_bytes, filetype="pdf") as doc:
                # 处理第一页（港口油价数据）
                if doc.page_count > 0:
                    bunker_df = self._process_page(doc[0], is_bunker=True)
                    if bunker_df is not None:
                        result['bunker'] = self._save_data(bunker_df, self.bunker_path, "BunkerPrices")

                # 处理第二页（燃料价格数据）
                if doc.page_count > 1:
                    fuel_df = self._process_page(doc[1], is_bunker=False)
                    if fuel_df is not None:
                        result['fuel'] = self._save_data(fuel_df, self.fuel_path, "FuelPrices")
        
        except Exception as e:
            logger.error(f"PDF处理失败: {str(e)}")
        
        return result

    def _process_page(self, page, is_bunker: bool) -> Optional[pd.DataFrame]:
        text = page.get_text().strip()
        date_match = re.search(DATE_PATTERN, text)
        if not date_match:
            return None
        
        date_str = date_match.group(1)
        try:
            date = datetime.strptime(date_str, "%B %d, %Y").date()
        except:
            return None

        columns = BUNKER_COLUMNS if is_bunker else FUEL_COLUMNS
        config = EXTRACTION_CONFIG[1] if is_bunker else EXTRACTION_CONFIG[2]
        
        data = {'Date': [date]}
        for pattern in config:
            matches = re.findall(pattern, text)
            for code, value in matches:
                if code in columns:
                    data[code] = [float(value)]
        
        # 填充缺失值为None
        for col in columns:
            if col != 'Date' and col not in data:
                data[col] = [None]
        
        return pd.DataFrame(data)

    def _save_data(self, new_df: pd.DataFrame, output_path: str, sheet_name: str) -> int:
        try:
            existing_df, exists = self.gh_manager.read_excel(output_path)
            if exists:
                combined_df = BunkerDataProcessor.merge_data(existing_df, new_df)
                combined_df = BunkerDataProcessor.clean_dataframe(combined_df)
            else:
                combined_df = new_df

            if self.gh_manager.save_excel(
                combined_df,
                output_path,
                f"Update {sheet_name} at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            ):
                return 1
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
        return 0

# --------------------------
# 缓存数据加载
# --------------------------
@st.cache_data(ttl=3600, show_spinner="加载历史数据...")
def load_history_data(path: str, columns: List[str]) -> pd.DataFrame:
    try:
        gh_manager = GitHubDataManager(
            st.secrets.github.token, 
            st.secrets.github.repo
        )
        df, exists = gh_manager.read_excel(path)
        if exists:
            df = BunkerDataProcessor.clean_dataframe(df)
            # 确保所有列都存在
            for col in columns:
                if col not in df.columns:
                    df[col] = None
            return df[columns]
    except Exception as e:
        logger.error(f"数据加载失败: {path} - {str(e)}")
    return pd.DataFrame(columns=columns)

# --------------------------
# Streamlit 界面
# --------------------------
def main_ui():
    st.set_page_config(page_title="船燃价格分析系统", layout="wide")
    st.title("Mariners' Bunker Price Analysis System")

    BUNKER_PATH = "data/bunker_prices.xlsx"
    FUEL_PATH = "data/fuel_prices.xlsx"

    # 文件上传区域
    with st.expander("📤 第一步 - 上传PDF报告", expanded=True):
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
                    pdf_bytes = file.getvalue()
                    extractor = EnhancedBunkerPriceExtractor(pdf_bytes, BUNKER_PATH, FUEL_PATH)
                    result = extractor.process_pdf()
                    
                    if result['bunker'] > 0 or result['fuel'] > 0:
                        st.toast(f"✅ {file.name} 处理成功（+{result['bunker']}油价/+{result['fuel']}燃料）")
                    else:
                        st.toast(f"⚠️ {file.name} 无新数据（可能为重复文件）")
                except Exception as e:
                    st.toast(f"❌ {file.name} 处理失败: {str(e)}")
        
            status.update(label="处理完成！", state="complete")
            st.cache_data.clear()

    # 加载数据
    bunker_df = load_history_data(BUNKER_PATH, BUNKER_COLUMNS)
    fuel_df = load_history_data(FUEL_PATH, FUEL_COLUMNS)

    # 创建标签页
    tab2, tab3, tab4, tab5 = st.tabs([
        "区域油价数据", "油价趋势分析", "燃料价格分析", "数据对比"
    ])

    # 区域油价数据
    with tab2:
        if not bunker_df.empty:
            st.subheader("区域油价数据（最新十日）")
            
            # 按区域分组
            regions = {
                "Asia Pacific/Middle East": ["MFSPD00", "MFFJD00", "MFJPD00", "BAMFB00", "MFSKD00", 
                                            "WKMFA00", "MFHKD00", "MFSHD00", "MFZSD00", "MFDSY00", 
                                            "MFDMB00", "MFDKW00", "MFDKF00", "MFDMM00", "MFDCL00"],
                "Europe": ["MFAGD00", "MFDBD00", "MFGBD00", "MFMLD00", "MFPRD00", "MFRDD00", 
                          "MFDAN00", "MFDGT00", "MFDHB00", "MFDIS00", "MFDLP00", "MFDNV00", 
                          "MFDPT00", "MFLIS00", "MFLOM00"],
                "Americas": ["MFHOD00", "MFNYD00", "MFLAD00", "MFNOD00", "MFPAD00", "MFSED00", 
                            "MFVAD00", "MFBAD00", "MFCRD00", "MFSAD00", "AMFVA00", "AMFCA00", 
                            "AMFGY00", "AMFLB00", "AMFMT00", "AMFSF00", "AMFMO00"]
            }
            
            for region, ports in regions.items():
                # 过滤实际存在的端口
                available_ports = [p for p in ports if p in bunker_df.columns]
                if not available_ports:
                    continue
                    
                st.subheader(f"{region} 油价数据")
                df_display = bunker_df[['Date'] + available_ports].copy()
                df_display = df_display.sort_values('Date', ascending=False).head(10)
                df_display = df_display.set_index("Date")
                
                # 重命名列显示
                df_display = df_display.rename(columns=lambda x: f"{PORT_CODE_MAPPING.get(x, x)} ({x})")
                st.dataframe(df_display, use_container_width=True, height=400)
                
                # 数据下载
                st.subheader(f"{region} 数据下载")
                col1, col2 = st.columns(2)
                with col1:
                    selected_date = st.selectbox(f"选择日期（{region}）", 
                                                options=df_display.index.astype(str).unique())
                    if selected_date:
                        daily_data = bunker_df[bunker_df['Date'].astype(str) == selected_date]
                        daily_data = daily_data[['Date'] + available_ports]
                        daily_data = daily_data.rename(columns=lambda x: f"{PORT_CODE_MAPPING.get(x, x)} ({x})")
                        st.download_button(
                            label=f"下载当日数据（{region}）",
                            data=daily_data.to_csv(index=False).encode('utf-8'),
                            file_name=f"{region.lower().replace(' ', '_')}_{selected_date}.csv",
                            mime="text/csv"
                        )
                with col2:
                    region_data = bunker_df[['Date'] + available_ports].copy()
                    region_data = region_data.rename(columns=lambda x: f"{PORT_CODE_MAPPING.get(x, x)} ({x})")
                    st.download_button(
                        label=f"下载完整数据（{region}）",
                        data=region_data.to_csv(index=False).encode('utf-8'),
                        file_name=f"{region.lower().replace(' ', '_')}_full.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("暂无数据")

    # 油价趋势分析
    with tab3:
        if not bunker_df.empty:
            st.subheader("多港口趋势比较")
            
            # 可用端口
            available_ports = [p for p in bunker_df.columns if p != 'Date']
            default_ports = [port for port in COMPARE_PORT_CODES if port in available_ports]
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_ports = st.multiselect(
                    "选择比较港口",
                    available_ports,
                    default=default_ports
                )
            with col2:
                years = sorted(bunker_df['Date'].dt.year.unique(), reverse=True)
                selected_year = st.selectbox("选择年份", years)
            
            if selected_ports:
                # 过滤数据
                filtered_df = bunker_df[bunker_df['Date'].dt.year == selected_year]
                filtered_df = filtered_df[['Date'] + selected_ports]
                
                # 创建图表
                fig = go.Figure()
                for port in selected_ports:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df[port],
                        mode='lines+markers',
                        name=f"{PORT_CODE_MAPPING.get(port, port)} ({port})",
                        connectgaps=True
                    ))
                
                fig.update_layout(
                    title=f"{selected_year}年燃油价格趋势",
                    xaxis_title="日期",
                    yaxis_title="价格 (USD)",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("请选择至少一个港口进行比较。")
        else:
            st.warning("暂无油价数据可供分析。")

    # 燃料价格分析
    with tab4:
        if not fuel_df.empty:
            st.subheader("替代燃料价格趋势")
            
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
    with tab5:
        if not bunker_df.empty:
            st.subheader("指定日期港口价格对比")
            
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

# --------------------------
# 主程序入口
# --------------------------
if __name__ == "__main__":
    main_ui()

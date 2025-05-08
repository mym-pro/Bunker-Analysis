import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import os
import re
import logging
import tempfile
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from github import Github
import base64
from io import BytesIO

# --------------------------
# 配置日志系统
# --------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# --------------------------
# 常量定义
# --------------------------
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

TAB1_COLUMN_ORDER = [
    "AMFSA00", "MFSPD00", "PPXDK00", "PUAFT00", "AAXYO00", "PUMFD00", "MFRDD00", "PUABC00", "PUAFN00",
    "AARTG00", "MFSAD00", "AAXWO00", "MFZSD00", "BFDZA00", "MGZSD00", "MFHKD00", "PUAER00", "AAXYQ00",
    "MFSKD00", "PUAGQ00", "AAXYS00", "MFSHD00", "AARKD00", "AAXYR00", "MFGBD00", "AAKAB00", "AARSU00",
    "MFNOD00", "AAGQE00", "AAWYA00", "AMFFA00", "MFFJD00", "PUAXP00", "AAXYP00"
]

REGION_ORDER = {
    "Asia Pacific/Middle East": [
        "MFSPD00", "MFFJD00", "MFJPD00", "BAMFB00", "MFSKD00", "WKMFA00", "MFHKD00",
        "MFSHD00", "MFZSD00", "MFDSY00", "MFDMB00", "MFDKW00", "MFDKF00", "MFDMM00", "MFDCL00"
    ],
    "Europe": [
        "MFAGD00", "MFDBD00", "MFGBD00", "MFMLD00", "MFPRD00", "MFRDD00", "MFDAN00",
        "MFDGT00", "MFDHB00", "MFDIS00", "MFDLP00", "MFDNV00", "MFDPT00", "MFLIS00", "MFLOM00"
    ],
    "Americas": [
        "MFHOD00", "MFNYD00", "MFLAD00", "MFNOD00", "MFPAD00", "MFSED00", "MFVAD00",
        "MFBAD00", "MFCRD00", "MFSAD00", "AMFVA00", "AMFCA00", "AMFGY00", "AMFLB00",
        "AMFMT00", "AMFSF00", "AMFMO00"
    ]
}
COMPARE_PORTS = ["Singapore", "Rotterdam", "Hong Kong", "Santos", "Zhoushan"]
FUEL_TYPES = ["MLBSO00", "LNBSF00"]

class BunkerDataProcessor:
    @staticmethod
    def format_date(date_series: pd.Series) -> pd.Series:
        return pd.to_datetime(date_series, errors='coerce').dt.date

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if 'Date' in df.columns:
            df['Date'] = BunkerDataProcessor.format_date(df['Date'])
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date', ascending=False)
            df = df.drop_duplicates(subset='Date', keep='first')
        return df.sort_values('Date', ascending=True).reset_index(drop=True)

    @staticmethod
    def merge_data(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        existing_df = BunkerDataProcessor.clean_dataframe(existing_df)
        new_df = BunkerDataProcessor.clean_dataframe(new_df)
        if existing_df.empty:
            return new_df, True
        combined = pd.concat([existing_df, new_df])
        return BunkerDataProcessor.clean_dataframe(combined), True

class GitHubDataManager:
    def __init__(self, token: str, repo_name: str):
        self.token = token
        self.repo_name = repo_name
        self.g = Github(self.token)
        self.repo = self.g.get_repo(self.repo_name)
    
    def read_excel(self, file_path: str) -> Tuple[pd.DataFrame, bool]:
        try:
            contents = self.repo.get_contents(file_path)
            return pd.read_excel(BytesIO(base64.b64decode(contents.content)), sheet_name=0), True
        except Exception as e:
            return pd.DataFrame(), False
    
    def save_excel(self, df: pd.DataFrame, file_path: str, commit_msg: str) -> bool:
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            content = output.getvalue()
            try:
                contents = self.repo.get_contents(file_path)
                self.repo.update_file(contents.path, commit_msg, content, contents.sha)
            except:
                self.repo.create_file(file_path, commit_msg, content)
            return True
        except Exception as e:
            logger.error(f"GitHub保存失败: {str(e)}")
            return False

class EnhancedBunkerPriceExtractor:
    def __init__(self, pdf_path: str, bunker_path: str, fuel_path: str):
        self.pdf_path = pdf_path
        self.bunker_path = bunker_path
        self.fuel_path = fuel_path

    def process_pdf(self) -> Dict[str, int]:
        try:
            doc = fitz.open(self.pdf_path)
            result = {'bunker': 0, 'fuel': 0}
            
            # 处理第一页（港口油价数据）
            bunker_df = self._process_page_1(doc[0])
            if bunker_df is not None:
                success = self._save_data(bunker_df, self.bunker_path, "BunkerPrices")
                result['bunker'] = 1 if success else 0

            # 处理第二页（燃料价格数据）
            fuel_df = self._process_page_2(doc[1])
            if fuel_df is not None:
                success = self._save_data(fuel_df, self.fuel_path, "FuelPrices")
                result['fuel'] = 1 if success else 0

            doc.close()
            return result
        except Exception as e:
            logger.error(f"PDF处理失败: {str(e)}")
            return {'bunker': 0, 'fuel': 0}

    def _process_page_1(self, page) -> Optional[pd.DataFrame]:
        coord_config = {'start_key': 'Bunkerwire', 'end_key': 'Ex-Wharf'}
        coords = self._get_page_coordinates(page, coord_config)
        if not coords:
            return None

        raw_text = self._extract_text_from_area(page, coords)
        date = self._extract_date(raw_text)
        if not date:
            return None

        pattern = re.compile(r"([A-Za-z\s\(\)-,]+)\s+([A-Z0-9]+)\s+(NA|\d+\.\d+)\s+(NANA|[+-]?\d+\.\d+)")
        matches = pattern.findall(raw_text.replace("\n", " "))
        if not matches:
            return None

        # 初始化所有需要的列
        all_columns = ['Date'] + [col for region in REGION_ORDER.values() for col in region]
        data = {col: [None] for col in all_columns}
        data['Date'] = [date]

        for port, code, price, _ in matches:
            if price != 'NA' and code in PORT_CODE_MAPPING:
                try:
                    data[code] = [float(price)]
                except ValueError:
                    continue

        df = pd.DataFrame(data)
        if len(df) > 1:
            logger.debug(f"Extracted DataFrame columns: {df.columns}")
            return df
        return None

    def _process_page_2(self, page) -> Optional[pd.DataFrame]:
        coord_config = {'start_key': 'Alternative marine fuels', 'end_key': 'Arab Gulf'}
        coords = self._get_page_coordinates(page, coord_config)
        if not coords:
            return None

        raw_text = self._extract_text_from_area(page, coords)
        date = self._extract_date(raw_text)
        if not date:
            return None

        pattern = re.compile(r"(MLBSO00|LNBSF00)\s+(\d+\.\d+|NA)")
        matches = pattern.findall(raw_text)
        if not matches:
            return None

        data = {'Date': [date]}
        for code, value in matches:
            if value != 'NA':
                try:
                    data[code] = [float(value)]
                except ValueError:
                    continue
        return pd.DataFrame(data) if len(data) > 1 else None

    def _get_page_coordinates(self, page, config: Dict) -> Optional[Dict]:
        blocks = page.get_text("blocks")
        coords = {'start_y': None, 'end_y': None}
        for block in blocks:
            text = block[4].strip()
            if config['start_key'] in text and not coords['start_y']:
                coords['start_y'] = block[1]
            if config['end_key'] in text and not coords['end_y']:
                coords['end_y'] = block[1]
        if None in [coords['start_y'], coords['end_y']]:
            return None
        return coords

    def _extract_text_from_area(self, page, coords: Dict) -> str:
        rect = fitz.Rect(
            0,
            min(coords['start_y'], coords['end_y']),
            page.rect.width,
            max(coords['start_y'], coords['end_y'])
        )
        return page.get_text("text", clip=rect)

    def _extract_date(self, text: str) -> Optional[datetime.date]:
        date_pattern = r"Volume\s+\d+\s+/\s+Issue\s+\d+\s+/\s+(\w+\s+\d{1,2},\s+\d{4})"
        match = re.search(date_pattern, text)
        if not match:
            return None
        try:
            return datetime.strptime(match.group(1), "%B %d, %Y").date()
        except Exception as e:
            logger.error(f"日期解析失败: {str(e)}")
            return None

    def _save_data(self, new_df: pd.DataFrame, output_path: str, sheet_name: str) -> bool:
        try:
            github_token = st.secrets.github.token
            repo_name = st.secrets.github.repo
            gh_manager = GitHubDataManager(github_token, repo_name)
            
            existing_df, exists = gh_manager.read_excel(output_path)
            if exists and not existing_df.empty:
                # 检查日期是否已存在
                if new_df['Date'].iloc[0] in existing_df['Date'].values:
                    # 如果日期已存在，覆盖该日期的数据
                    existing_df = existing_df[existing_df['Date'] != new_df['Date'].iloc[0]]
                combined_df = pd.concat([existing_df, new_df])
                combined_df = BunkerDataProcessor.clean_dataframe(combined_df)
            else:
                combined_df = new_df

            return gh_manager.save_excel(
                combined_df,
                output_path,
                f"Update {sheet_name} at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            )
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
            return False

@st.cache_data(ttl=3600, show_spinner="加载历史数据...")
def load_history_data(path: str) -> pd.DataFrame:
    try:
        github_token = st.secrets.github.token
        repo_name = st.secrets.github.repo
        gh_manager = GitHubDataManager(github_token, repo_name)
        df, exists = gh_manager.read_excel(path)
        if exists:
            # 确保所有需要的列都存在
            all_columns = ['Date'] + [col for region in REGION_ORDER.values() for col in region]
            df = ensure_columns(df, all_columns)
            return BunkerDataProcessor.clean_dataframe(df)
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"数据加载失败: {path} - {str(e)}")
        return pd.DataFrame()

def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    missing_columns = [col for col in columns if col not in df.columns]
    for col in missing_columns:
        df[col] = pd.NA
    return df

def generate_excel_download(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.getvalue()

def main_ui():
    st.set_page_config(page_title="船燃价格分析系统", layout="wide")
    st.title("Mariners' Bunker Price Analysis System")

    BUNKER_PATH = "data/bunker_prices.xlsx"
    FUEL_PATH = "data/fuel_prices.xlsx"

    with st.expander("📤 第一步 - 上传PDF报告", expanded=True):
        uploaded_files = st.file_uploader(
            "选择Bunkerwire PDF报告（支持多选）",
            type=["pdf"],
            accept_multiple_files=True
        )

    if uploaded_files:
        with st.status("正在解析文件...", expanded=True) as status:
            for file in uploaded_files:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getbuffer())
                        tmp_path = tmp.name

                    extractor = EnhancedBunkerPriceExtractor(tmp_path, BUNKER_PATH, FUEL_PATH)
                    result = extractor.process_pdf()
                    
                    if result['bunker'] > 0 or result['fuel'] > 0:
                        st.toast(f"✅ {file.name} 处理成功（+{result['bunker']}油价/+{result['fuel']}燃料）")
                    else:
                        st.toast(f"⚠️ {file.name} 无新数据（可能为重复文件）")

                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception as e:
                    st.toast(f"❌ {file.name} 处理失败: {str(e)}")
        
            status.update(label="处理完成！", state="complete")
            st.cache_data.clear()

    bunker_df = load_history_data(BUNKER_PATH)
    fuel_df = load_history_data(FUEL_PATH)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Bunker Wire", "区域油价数据", "油价趋势分析", "燃料价格分析", "数据对比"
    ])

    with tab1:
        if not bunker_df.empty:
            st.subheader("Bunker Wire数据（最新十日）")
            df_display = bunker_df.copy().sort_values('Date', ascending=False).head(10)
            tab1_columns = [col for col in TAB1_COLUMN_ORDER if col in df_display.columns]
            df_display = df_display[['Date'] + tab1_columns]
            df_display = df_display.rename(columns=lambda x: f"{PORT_CODE_MAPPING.get(x, x)} ({x})" if x != "Date" else x)
            st.dataframe(df_display.set_index("Date"), use_container_width=True, height=400)
            
            st.subheader("数据下载")
            col1, col2 = st.columns(2)
            with col1:
                selected_date = st.selectbox("选择日期（Bunker Wire）", options=df_display['Date'].astype(str).unique())
                if selected_date:
                    daily_data = bunker_df[bunker_df['Date'].astype(str) == selected_date][['Date'] + tab1_columns]
                    st.download_button(
                        label="下载当日数据",
                        data=generate_excel_download(daily_data),
                        file_name=f"bunker_wire_{selected_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            with col2:
                st.download_button(
                    label="下载完整数据",
                    data=generate_excel_download(bunker_df[['Date'] + tab1_columns]),
                    file_name="bunker_wire_full.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("暂无数据")

    with tab2:
        if not bunker_df.empty:
            st.subheader("区域油价数据（最新十日）")
            region_order = []
            for region in ["Asia Pacific/Middle East", "Europe", "Americas"]:
                region_order.extend(REGION_ORDER[region])
            
            # 确保所有需要的列都存在
            bunker_df = ensure_columns(bunker_df, region_order)
            
            df_display = bunker_df.copy().sort_values('Date', ascending=False).head(10)
            df_display = df_display[['Date'] + region_order]
            df_display = df_display.rename(columns=lambda x: f"{PORT_CODE_MAPPING.get(x, x)} ({x})" if x != "Date" else x)
            st.dataframe(df_display.set_index("Date"), use_container_width=True, height=400)
            
            st.subheader("数据下载")
            col1, col2 = st.columns(2)
            with col1:
                selected_date = st.selectbox("选择日期（区域数据）", options=df_display['Date'].astype(str).unique())
                if selected_date:
                    daily_data = bunker_df[bunker_df['Date'].astype(str) == selected_date][['Date'] + region_order]
                    st.download_button(
                        label="下载当日数据",
                        data=generate_excel_download(daily_data),
                        file_name=f"regional_{selected_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            with col2:
                st.download_button(
                    label="下载完整数据",
                    data=generate_excel_download(bunker_df[['Date'] + region_order]),
                    file_name="regional_full.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("暂无数据")

    with tab3:
        if not bunker_df.empty:
            st.subheader("Multi-Port Trend Comparison")
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_ports = st.multiselect(
                    "Select Ports for Comparison",
                    [p for p in bunker_df.columns if p != 'Date'],
                    default=COMPARE_PORTS[:2]
                )
            with col2:
                selected_year = st.selectbox(
                    "Select Year",
                    sorted(bunker_df['Date'].apply(lambda x: x.year).unique(), reverse=True)
                )
            filtered_df = bunker_df.loc[bunker_df['Date'].apply(lambda x: x.year) == selected_year]
            if selected_ports:
                fig = go.Figure()
                for port in selected_ports:
                    if port in filtered_df.columns:
                        fig.add_trace(go.Scatter(
                            x=filtered_df['Date'],
                            y=filtered_df[port],
                            mode='lines+markers',
                            name=port,
                            connectgaps=True
                        ))
                fig.update_layout(
                    title=f"Fuel Price Trends in {selected_year}",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("请选择至少一个港口进行比较。")
        else:
            st.warning("暂无油价数据可供分析。")

    with tab4:
        if not fuel_df.empty:
            st.subheader("替代燃料价格趋势")
            fuel_cols = ['Date'] + [col for col in FUEL_TYPES if col in fuel_df.columns]
            ordered_fuel_df = fuel_df[fuel_cols]
            
            st.dataframe(
                ordered_fuel_df.sort_values('Date', ascending=True).tail(10).set_index("Date"),
                use_container_width=True
            )
            
            fig = go.Figure()
            for fuel_type in FUEL_TYPES:
                if fuel_type in fuel_df.columns:
                    fig.add_trace(go.Scatter(
                        x=fuel_df['Date'],
                        y=fuel_df[fuel_type],
                        name=fuel_type,
                        mode='lines+markers',
                        connectgaps=True
                    ))
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        if not bunker_df.empty:
            st.subheader("指定日期港口价格对比")
            date_options = sorted(bunker_df['Date'].astype(str).unique(), reverse=True)
            col1, col2 = st.columns(2)
            with col1:
                date1 = st.selectbox("选择对比日期1", date_options)
            with col2:
                date2 = st.selectbox("选择对比日期2", date_options)
            if date1 and date2:
                df1 = bunker_df.loc[bunker_df['Date'].astype(str) == date1]
                df2 = bunker_df.loc[bunker_df['Date'].astype(str) == date2]
                if not df1.empty and not df2.empty:
                    comparison = []
                    for port in COMPARE_PORTS:
                        if port in df1.columns and port in df2.columns:
                            price1 = df1[port].values[0] if not df1[port].isna().all() else None
                            price2 = df2[port].values[0] if not df2[port].isna().all() else None
                            if price1 is not None and price2 is not None:
                                change = ((price1 - price2) / price2 * 100) if price2 != 0 else None
                            else:
                                change = None
                            comparison.append({
                                "Port": port,
                                date1: price1,
                                date2: price2,
                                "Change (%)": f"{change:.2f}%" if change is not None else "N/A"
                            })
                    if comparison:
                        st.dataframe(
                            pd.DataFrame(comparison).set_index("Port"),
                            use_container_width=True,
                            hide_index=False
                        )
                    else:
                        st.warning("未找到选定日期的数据或选定港口的数据不完整。")
                else:
                    st.warning("未找到选定日期的数据。")
        else:
            st.warning("暂无油价数据可供对比。")

if __name__ == "__main__":
    main_ui()

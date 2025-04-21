import streamlit as st
import fitz
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

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 常量定义
REGION_PORTS = {
    "Asia Pacific/Middle East": ["Singapore", "Fujairah", "Japan", "West Japan", "South Korea", 
                                 "South Korea (West)", "Hong Kong", "Shanghai", "Zhoushan", 
                                 "Sydney", "Melbourne", "Kuwait", "Khor Fakkan", "Mumbai", "Colombo"],
    "Europe": ["Algeciras", "Durban", "Gibraltar", "Malta", "Piraeus", "Rotterdam", 
               "Antwerp", "Gothenburg", "Hamburg", "Istanbul", "Las Palmas", "Novorossiisk",
               "St. Petersburg", "Lisbon", "Lome"],
    "Americas": ["Houston", "New York", "Los Angeles", "New Orleans", "Philadelphia", 
                 "Seattle", "Vancouver", "Buenos Aires", "Cartagena", "Santos", 
                 "Valparaiso", "Callao", "Guayaquil", "La Libertad", "Montevideo", 
                 "San Francisco", "Montreal*"]
}

COMPARE_PORTS = ["Singapore", "Rotterdam", "Hong Kong", "Santos", "Zhoushan"]
FUEL_TYPES = ["MLBSO00", "LNBSF00"]

# 辅助函数
def format_date(date_series):
    """将日期列格式化为 YYYY-MM-DD"""
    return pd.to_datetime(date_series).dt.date


def clean_dataframe(df):
    """清洗数据：转换数值列，去除全为空的列"""
    df = df.replace('NA', pd.NA).dropna(how='all', axis=1)
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 将无法转换的值设为 NaN
    return df


def merge_and_sort_data(existing_df, new_df):
    """合并现有数据和新数据，并按日期排序"""
    existing_dates = set(existing_df['Date'])
    new_data = new_df[~new_df['Date'].isin(existing_dates)]
    if new_data.empty:
        return existing_df, False
    combined_df = pd.concat([existing_df, new_data]).drop_duplicates()
    return combined_df.sort_values('Date').reset_index(drop=True), True


class EnhancedBunkerPriceExtractor:
    def __init__(self, pdf_path, output_excel_paths):
        self.pdf_path = pdf_path
        self.output_excel_paths = output_excel_paths
        self.logger = logger
        self._validate_paths()

    def _validate_paths(self):
        """路径验证"""
        for path in self.output_excel_paths:
            if not isinstance(path, Path):
                raise ValueError(f"无效路径格式: {path}")

    def extract_structured_bunker_prices(self):
        """增强的数据提取方法"""
        try:
            doc = fitz.open(self.pdf_path)
            date = None
            new_data_count = 0

            # 处理第一页
            df_page_1 = self._process_page_1(doc[0])
            if df_page_1 is not None:
                if self._save_data(df_page_1, self.output_excel_paths[0], "BunkerPrices"):
                    new_data_count += 1

            # 处理第二页
            df_page_2 = self._process_page_2(doc[1], df_page_1['Date'].iloc[0] if df_page_1 is not None else None)
            if df_page_2 is not None:
                if self._save_data(df_page_2, self.output_excel_paths[1], "FuelPrices"):
                    new_data_count += 1

            doc.close()
            return new_data_count
        except Exception as e:
            self.logger.error(f"处理PDF时发生错误: {str(e)}")
            return 0

    def _get_key_coordinates(self, page, start_keyword, end_keyword, right_boundary_keyword=None, left_boundary_keyword=None):
        """获取关键区域坐标"""
        blocks = page.get_text("blocks")
        coords = {
            "start_y": None,
            "end_y": None,
            "left_x": 0,
            "right_x": page.rect.width
        }

        for block in blocks:
            text = block[4].strip()
            if start_keyword in text:
                coords["start_y"] = block[1]
            if end_keyword in text:
                coords["end_y"] = block[1]
            if right_boundary_keyword and right_boundary_keyword in text:
                coords["right_x"] = block[0]
            if left_boundary_keyword and left_boundary_keyword in text:
                coords["left_x"] = block[2]

        if None in coords.values():
            self.logger.error("关键坐标定位失败")
            return None
        return coords

    def _extract_raw_text(self, page, coords):
        """提取指定区域的原始文本"""
        table_rect = fitz.Rect(
            coords["left_x"],
            min(coords["start_y"], coords["end_y"]),
            coords["right_x"],
            max(coords["start_y"], coords["end_y"])
        )
        return page.get_text("text", clip=table_rect)

    def _process_page_1(self, page):
        """处理第一页内容并提取日期"""
        coords = self._get_key_coordinates(page, "Bunkerwire", "Ex-Wharf", "Marine Fuel (PGB page 30)")
        if not coords:
            self.logger.error("第一页关键坐标定位失败")
            return None

        raw_text = self._extract_raw_text(page, coords)
        if not raw_text:
            self.logger.warning("第一页未提取到文本")
            return None

        date_pattern = r"Volume\s+\d+\s+/\s+Issue\s+\d+\s+/\s+(\w+\s+\d{1,2},\s+\d{4})"
        match = re.search(date_pattern, raw_text)
        if match:
            date = match.group(1).strip()
        else:
            self.logger.warning("第一页未匹配到日期信息。")
            return None

        try:
            date_obj = datetime.strptime(match.group(1), "%B %d, %Y")
            data = {"Date": date_obj.date()}  # 只保留日期部分
        except ValueError as e:
            self.logger.error(f"日期格式错误: {e}")
            return None

        pattern = re.compile(r"([A-Za-z\s\(\)-,]+)\s+([A-Z0-9]+)\s+(NA|\d+\.\d+)\s+(NANA|[+-]?\d+\.\d+)")
        start_marker = "Singapore"
        start_index = raw_text.find(start_marker)

        if start_index == -1:
            self.logger.warning("未找到起始标记 'Singapore'。")
            return None

        mid_relevant_text = raw_text[start_index:]
        relevant_text = mid_relevant_text.replace("\n", " ").replace("\t", " ")
        relevant_text = re.sub(r"\s+", " ", relevant_text).strip() 
        matches = pattern.findall(relevant_text)
        print(matches)

        if not matches:
            self.logger.warning("未匹配到任何数据。")
            return None

        for port, code, price, change in matches:
            data[port.strip()] = float(price) if price != 'NA' else None  # 如果是 'NA'，保存为 None
        return pd.DataFrame([data]).round(2)

    def _process_page_2(self, page, date):
        """处理第二页内容"""
        coords = self._get_key_coordinates(page, "Alternative marine fuels", "Arab Gulf")
        if not coords:
            self.logger.error("第二页坐标定位失败")
            return None

        raw_text = self._extract_raw_text(page, coords)

        if not raw_text:
            self.logger.warning("第二页未提取到文本")
            return None

        pattern = re.compile(r"(MLBSO00|LNBSF00)\s+(\d+\.\d+|NA)")
        matches = pattern.findall(raw_text)
        print(matches)
        if not matches:
            self.logger.warning("第二页未找到燃油数据")
            return None

        data = {"Date": date} if date else {}
        for code, value in matches:
            data[code.strip()] = float(value) if value != 'NA' else None  # 如果是 'NA'，保存为 None
        return pd.DataFrame([data]).round(2) if data else None

    def _save_data(self, df, output_path, sheet_name):
        """优化后的数据存储方法"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                temp_path = tmp.name

            # 数据清洗和验证
            df = clean_dataframe(df)
            df['Date'] = format_date(df['Date'])  # 确保日期格式一致

            # 合并现有数据（如果存在）
            if output_path.exists():
                existing_df = pd.read_excel(output_path)
                existing_df['Date'] = format_date(existing_df['Date'])  # 确保日期格式一致

                combined_df, new_data_added = merge_and_sort_data(existing_df, df)
                if not new_data_added:
                    self.logger.info("没有新数据需要添加。")
                    return False  # 没有新数据，直接返回
            else:
                combined_df = df

            # 使用pandas直接保存（提升性能）
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 原子操作替换文件
            os.replace(temp_path, output_path)
            self.logger.info(f"数据已成功保存到 {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"保存失败: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False


@st.cache_data(ttl=3600, show_spinner=False)
def load_history_data(path):
    """Optimized data loading with caching"""
    try:
        if path.exists():
            df = pd.read_excel(path)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date  # 转换为日期部分
            logger.info(f"成功加载历史数据: {path}")
            return df.sort_values('Date', ascending=False).reset_index(drop=True)  # 按日期降序排列
        else:
            logger.warning(f"文件不存在: {path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        return pd.DataFrame()


def show_timed_message(message, message_type="success", duration=3):
    """显示定时消息"""
    placeholder = st.empty()
    if message_type == "success":
        placeholder.success(message)
    else:
        placeholder.warning(message)
    time.sleep(duration)
    placeholder.empty()


def generate_excel_download(df, filename):
    """生成Excel下载链接"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.getvalue()


def main():
    st.set_page_config(page_title="Bunker Price Analysis", layout="wide")
    st.title("🌊 Marine Fuel Price Analysis System")

    history_dir = Path("history_data")
    history_dir.mkdir(exist_ok=True)
    bunker_path = history_dir / "bunker_prices_history.xlsx"
    fuel_path = history_dir / "fuel_prices_history.xlsx"

    # 文件上传处理
    with st.expander("📁 STEP 1 - 上传PDF文件", expanded=True):
        uploaded_files = st.file_uploader(
            "选择需要分析的Bunkerwire PDF文件", 
            type=["pdf"],
            accept_multiple_files=True,
            help="可同时上传多个PDF文件"
        )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(uploaded_file.getbuffer())
                    pdf_path = tmp_pdf.name

                extractor = EnhancedBunkerPriceExtractor(pdf_path, [bunker_path, fuel_path])
                new_data_count = extractor.extract_structured_bunker_prices()

                os.unlink(pdf_path)  # 删除临时文件

                if new_data_count > 0:
                    show_timed_message(f"✅ 成功添加 {new_data_count} 条新数据！")
                else:
                    show_timed_message("⚠️ 未添加新数据（可能是重复数据）", "warning")

            except Exception as e:
                st.error(f"处理文件 {uploaded_file.name} 时出错: {str(e)}")
        # 清理缓存
        st.cache_data.clear()

    # 数据展示模块
    bunker_df = load_history_data(bunker_path)
    fuel_df = load_history_data(fuel_path)

    with st.expander("📊 STEP 2 - 数据分析", expanded=True):
        tab1, tab2, tab3, tab4 = st.tabs(["港口油价信息", "油价趋势分析", "燃料价格分析", "数据对比"])

        with tab1:
            if not bunker_df.empty:
                st.subheader("近期油价趋势（最近10个记录）")
                recent_data = bunker_df.head(10)

                for region, ports in REGION_PORTS.items():
                    st.subheader(f"🏙️ {region}")
                    region_cols = [col for col in recent_data.columns if col in ports]
                    st.dataframe(
                        recent_data[["Date"] + region_cols].set_index("Date"),
                        use_container_width=True,
                        height=300
                    )

        with tab2:
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
                
                filtered_df = bunker_df[bunker_df['Date'].apply(lambda x: x.year) == selected_year]
                
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

        with tab3:
            if not fuel_df.empty:
                st.subheader("替代燃料价格趋势")
                
                st.dataframe(
                    fuel_df.head(10).set_index("Date"),
                    use_container_width=True
                )
                
                fig = go.Figure()
                for fuel_type in FUEL_TYPES:
                    fig.add_trace(go.Scatter(
                        x=fuel_df['Date'],
                        y=fuel_df[fuel_type],
                        name=fuel_type,
                        mode='lines+markers',
                        connectgaps=True
                    ))
                
                fig.update_layout(
                    height=600,
                    template="plotly_white",
                    yaxis_title="价格 (USD/吨)",
                    xaxis_title="日期",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("指定日期港口价格对比")
            if not bunker_df.empty:
                date_options = sorted(bunker_df['Date'].astype(str).unique(), reverse=True)

                col1, col2 = st.columns(2)
                with col1:
                    date1 = st.selectbox("选择对比日期1", date_options)
                with col2:
                    date2 = st.selectbox("选择对比日期2", date_options)

                if date1 and date2:
                    df1 = bunker_df[bunker_df['Date'].astype(str) == date1]
                    df2 = bunker_df[bunker_df['Date'].astype(str) == date2]

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
                                use_container_width=True
                            )
                        else:
                            st.warning("未找到选定日期的数据或选定港口的数据不完整。")
                    else:
                        st.warning("未找到选定日期的数据。")
            else:
                st.warning("暂无油价数据可供对比。")

    with st.expander("📥 STEP 3 - 数据下载", expanded=True):
        st.subheader("完整数据下载")
        col1, col2 = st.columns(2)
        with col1:
            if bunker_path.exists():
                with open(bunker_path, "rb") as f:
                    st.download_button(
                        label="下载完整油价数据",
                        data=f,
                        file_name="bunker_prices_full.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        with col2:
            if fuel_path.exists():
                with open(fuel_path, "rb") as f:
                    st.download_button(
                        label="下载完整燃料数据",
                        data=f,
                        file_name="alternative_fuels_full.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        st.subheader("单日数据下载")
        col1, col2 = st.columns(2)
        
        with col1:
            if not bunker_df.empty:
                selected_bunker_date = st.selectbox(
                    "选择油价日期",
                    options=sorted(bunker_df['Date'].astype(str).unique(), reverse=True)
                )
                if selected_bunker_date:
                    daily_bunker = bunker_df[bunker_df['Date'].astype(str) == selected_bunker_date]
                    st.download_button(
                        label="下载当日油价数据",
                        data=generate_excel_download(daily_bunker, "daily_bunker.xlsx"),
                        file_name=f"bunker_{selected_bunker_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            if not fuel_df.empty:
                selected_fuel_date = st.selectbox(
                    "选择燃料日期",
                    options=sorted(fuel_df['Date'].astype(str).unique(), reverse=True)
                )
                if selected_fuel_date:
                    daily_fuel = fuel_df[fuel_df['Date'].astype(str) == selected_fuel_date]
                    st.download_button(
                        label="下载当日燃料数据",
                        data=generate_excel_download(daily_fuel, "daily_fuel.xlsx"),
                        file_name=f"fuel_{selected_fuel_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

if __name__ == "__main__":
    main()

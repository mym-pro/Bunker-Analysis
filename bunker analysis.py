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


class BunkerDataProcessor:
    """数据处理工具类"""
    
    @staticmethod
    def format_date(date_series: pd.Series) -> pd.Series:
        """统一日期格式为YYYY-MM-DD"""
        return pd.to_datetime(date_series, errors='coerce').dt.date

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗：
        1. 转换数值列
        2. 去除全为空值的列
        3. 处理异常值
        """
        # 替换NA并去除全空列
        df = df.replace('NA', pd.NA).dropna(how='all', axis=1)
        
        # 数值类型转换
        for col in df.columns:
            if col != 'Date':
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce') # 非数值转为NaN
                
        # 去除日期列中的无效日期
        if 'Date' in df.columns:
            df['Date'] = BunkerDataProcessor.format_date(df['Date'])
            df = df.dropna(subset=['Date'])
            
        return df.dropna(how='all', axis=1)

    @staticmethod
    def merge_data(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """
        数据合并逻辑：
        1. 合并时自动去重
        2. 返回是否有新增数据的标志
        """
        if existing_df.empty:
            return new_df, True
            
        existing_dates = set(existing_df['Date'])
        new_data = new_df[~new_df['Date'].isin(existing_dates)]
        
        if new_data.empty:
            return existing_df, False
            
        return pd.concat([existing_df, new_data]).drop_duplicates().sort_values('Date').reset_index(drop=True), True

class GitHubDataManager:
    def __init__(self, token: str, repo_name: str):
        self.token = token
        self.repo_name = repo_name
        self.g = Github(self.token)
        self.repo = self.g.get_repo(self.repo_name)
    
    def read_excel(self, file_path: str) -> Tuple[pd.DataFrame, bool]:
        """修复返回值问题"""
        try:
            contents = self.repo.get_contents(file_path)
            return pd.read_excel(BytesIO(base64.b64decode(contents.content)), sheet_name=0), True
        except Exception as e:
            return pd.DataFrame(), False  # 明确返回元组
    
    def save_excel(self, df: pd.DataFrame, file_path: str, commit_msg: str) -> bool:
        """保存Excel文件到GitHub"""
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
BUNKER_PATH = "data/bunker_prices.xlsx"
FUEL_PATH = "data/fuel_prices.xlsx"
class EnhancedBunkerPriceExtractor:
    """增强版PDF数据提取器"""
    
    def __init__(self, pdf_path: str):
        """移除本地路径依赖"""
        self.pdf_path = pdf_path

    def _validate_paths(self):
        """路径验证"""
        for path in [self.bunker_path, self.fuel_path]:
            if not isinstance(path, Path):
                raise ValueError(f"无效路径格式: {path} 必须为Path对象")

    def process_pdf(self) -> Dict[str, int]:
        """
        主处理方法：
        返回包含新增数据数量的字典 {'bunker': 新增油价条目数, 'fuel': 新增燃料条目数}
        """
        try:
            doc = fitz.open(self.pdf_path)
            result = {'bunker': 0, 'fuel': 0}
            
            # 处理第一页（油价数据）
            bunker_df = self._process_page_1(doc[0])
            if bunker_df is not None:
                success = self._save_data(bunker_df, self.bunker_path, "BunkerPrices")
                result['bunker'] = 1 if success else 0

            # 处理第二页（燃料数据）
            fuel_df = self._process_page_2(doc[1], bunker_df['Date'].iloc[0] if bunker_df is not None else None)
            if fuel_df is not None:
                success = self._save_data(fuel_df, self.fuel_path, "FuelPrices")
                result['fuel'] = 1 if success else 0

            doc.close()
            return result
        except Exception as e:
            logger.error(f"PDF处理失败: {str(e)}")
            return {'bunker': 0, 'fuel': 0}

    def _get_page_coordinates(self, page, config: Dict) -> Optional[Dict]:
        """
        获取页面关键区域坐标
        config示例：
        {
            'start_key': '起始关键词',
            'end_key': '结束关键词',
            'right_boundary': '右侧边界关键词',  # 可选
            'left_boundary': '左侧边界关键词'   # 可选
        }
        """
        blocks = page.get_text("blocks")
        coords = {
            'start_y': None,
            'end_y': None,
            'left_x': 0,
            'right_x': page.rect.width
        }

        for block in blocks:
            text = block[4].strip()
            # 定位起始位置
            if config['start_key'] in text and coords['start_y'] is None:
                coords['start_y'] = block[1]
            # 定位结束位置
            if config['end_key'] in text and coords['end_y'] is None:
                coords['end_y'] = block[1]
            # 右侧边界
            if 'right_boundary' in config and config['right_boundary'] in text:
                coords['right_x'] = block[0]
            # 左侧边界
            if 'left_boundary' in config and config['left_boundary'] in text:
                coords['left_x'] = block[2]

        if None in [coords['start_y'], coords['end_y']]:
            logger.warning(f"页面坐标定位失败: {config}")
            return None
        return coords

    def _process_page_1(self, page) -> Optional[pd.DataFrame]:
        """处理油价页面"""
        # 定义页面解析配置
        coord_config = {
            'start_key': 'Bunkerwire',
            'end_key': 'Ex-Wharf',
            'right_boundary': 'Marine Fuel (PGB page 30)'
        }
        coords = self._get_page_coordinates(page, coord_config)
        if not coords:
            return None

        # 提取原始文本
        raw_text = self._extract_text_from_area(page, coords)
        if not raw_text:
            return None

        # 提取日期信息
        date = self._extract_date(raw_text)
        if not date:
            return None

        # 使用优化后的正则表达式提取数据
        pattern = re.compile(r"([A-Za-z\s\(\)-,]+)\s+([A-Z0-9]+)\s+(NA|\d+\.\d+)\s+(NANA|[+-]?\d+\.\d+)")
        start_marker = "Singapore"
        start_index = raw_text.find(start_marker)

        if start_index == -1:
            logger.warning("未找到起始标记 'Singapore'。")
            return None

        mid_relevant_text = raw_text[start_index:]
        relevant_text = mid_relevant_text.replace("\n", " ").replace("\t", " ")
        relevant_text = re.sub(r"\s+", " ", relevant_text).strip() 
        matches = pattern.findall(relevant_text)
        logger.debug(f"Page 1 matches: {matches}")
        if not matches:
            logger.warning("未在页面1找到有效数据")
            return None

        # 构建DataFrame
        data = {'Date': [date]}
        for port, code, price, change in matches:
            # 只保留有效的价格数据
            if price != 'NA':
                data[port.strip()] = [float(price)]
        return pd.DataFrame(data) if len(data) > 1 else None

    def _process_page_2(self, page, date) -> Optional[pd.DataFrame]:
        """处理燃料价格页面"""
        coord_config = {
            'start_key': 'Alternative marine fuels',
            'end_key': 'Arab Gulf'
        }
        coords = self._get_page_coordinates(page, coord_config)
        if not coords:
            return None

        raw_text = self._extract_text_from_area(page, coords)
        if not raw_text:
            return None

        # 提取燃料数据
        pattern = re.compile(r"(MLBSO00|LNBSF00)\s+(\d+\.\d+|NA)")
        matches = pattern.findall(raw_text)
        logger.debug(f"Page 2 matches: {matches}")
        if not matches:
            logger.warning("未在页面2找到有效数据")
            return None

        data = {'Date': [date]} if date else {}
        for code, value in matches:
            if value != 'NA':
                data[code] = [float(value)]
        return pd.DataFrame(data) if len(data) > 1 else None

    def _extract_text_from_area(self, page, coords: Dict) -> str:
        """从指定区域提取文本"""
        rect = fitz.Rect(
            coords['left_x'],
            min(coords['start_y'], coords['end_y']),
            coords['right_x'],
            max(coords['start_y'], coords['end_y'])
        )
        return page.get_text("text", clip=rect)

    def _extract_date(self, text: str) -> Optional[datetime.date]:
        """从文本中提取日期"""
        date_pattern = r"Volume\s+\d+\s+/\s+Issue\s+\d+\s+/\s+(\w+\s+\d{1,2},\s+\d{4})"
        match = re.search(date_pattern, text)
        if not match:
            return None
        try:
            return datetime.strptime(match.group(1), "%B %d, %Y").date()
        except ValueError:
            return None

    def _save_data(self, new_df: pd.DataFrame, output_path: str, sheet_name: str) -> bool:
        # 添加列对齐逻辑
        existing_df, exists = gh_manager.read_excel(output_path)
        if exists and not existing_df.empty:
            # 处理列不一致问题
            all_columns = list(set(existing_df.columns) | set(new_df.columns))
            existing_df = existing_df.reindex(columns=all_columns, fill_value=pd.NA)
            new_df = new_df.reindex(columns=all_columns, fill_value=pd.NA)
            combined_df = pd.concat([existing_df, new_df])
        else:
            combined_df = new_df
            
            # 保存数据
            return gh_manager.save_excel(
                combined_df,
                output_path,
                f"Update {sheet_name} at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
            return False


# --------------------------
# Streamlit界面组件
# --------------------------
@st.cache_data(ttl=3600, show_spinner="加载历史数据...")
def load_history_data(path: str) -> pd.DataFrame:
    """修改后的数据加载函数"""
    try:
        github_token = st.secrets.github.token
        repo_name = st.secrets.github.repo
        gh_manager = GitHubDataManager(github_token, repo_name)
        
        df, exists = gh_manager.read_excel(path)
        if exists:
            df['Date'] = BunkerDataProcessor.format_date(df['Date'])
            return df.sort_values('Date', ascending=False).reset_index(drop=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"数据加载失败: {path} - {str(e)}")
        return pd.DataFrame()

def show_status(message: str, message_type: str = "success", duration: int = 3):
    """显示状态提示信息"""
    if message_type == "success":
        st.toast(f"✅ {message}", icon="✅")
    elif message_type == "warning":
        st.toast(f"⚠️ {message}", icon="⚠️")
    else:
        st.toast(f"❌ {message}", icon="❌")
    time.sleep(duration)

def generate_excel_download(df: pd.DataFrame) -> bytes:
    """生成Excel文件字节流（添加空数据校验）"""
    if df.empty:
        raise ValueError("无法导出空数据集")
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.getvalue()

def on_download_click(success: bool, filename: str):
    """下载回调函数"""
    if success:
        st.toast(f"✅ {filename} 下载已开始，请查看浏览器下载列表")
    else:
        st.toast(f"⚠️ 下载文件为空，未生成下载", icon="⚠️")
def main_ui():
    """主界面布局"""
    st.set_page_config(page_title="船燃价格分析系统", layout="wide")
    st.title(" Mariners' Bunker Price Analysis System ")

    # 初始化数据存储路径
    BUNKER_PATH = "data/bunker_prices.xlsx"
    FUEL_PATH = "data/fuel_prices.xlsx"

    # 初始化session状态
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'show_final_toast' not in st.session_state:
        st.session_state.show_final_toast = False

    # --------------------------
    # 文件上传模块
    # --------------------------
    with st.expander("📤 第一步 - 上传PDF报告", expanded=True):
        uploaded_files = st.file_uploader(
            "选择Bunkerwire PDF报告（支持多选）",
            type=["pdf"],
            accept_multiple_files=True,
            help="请上传最新版的Bunkerwire PDF文件"
        )

    # 只处理新上传的文件
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    # --------------------------
    # 数据处理逻辑
    # --------------------------
    total_added = {'bunker': 0, 'fuel': 0}
    error_messages = []
    
    if new_files:
        with st.status("正在解析文件...", expanded=True) as status:
            for file in new_files:
                try:
                    # ...原有代码...
                except Exception as e:
                    # 添加详细错误日志
                    logger.error(f"文件处理错误详情: {traceback.format_exc()}")
                    error_messages.append(f"❌ {file.name} 处理失败: {str(e)}（详细日志请查看控制台）")
                    # 创建临时PDF文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getbuffer())
                        pdf_path = tmp.name

                    # 处理PDF
                    extractor = EnhancedBunkerPriceExtractor(pdf_path, [bunker_path, fuel_path])
                    result = extractor.process_pdf()
                    
                    # 记录处理结果
                    if result['bunker'] > 0 or result['fuel'] > 0:
                        st.session_state.processed_files.add(file.name)
                        total_added['bunker'] += result['bunker']
                        total_added['fuel'] += result['fuel']
                        st.toast(f"✅ {file.name} 处理成功（+{result['bunker']}油价/+{result['fuel']}燃料）")
                    else:
                        st.toast(f"⚠️ {file.name} 无新数据（可能为重复文件）")

                    # 清理临时文件
                    os.unlink(pdf_path)

                except Exception as e:
                    error_messages.append(f"❌ {file.name} 处理失败: {str(e)}")
                    logger.error(f"文件处理错误: {file.name} - {str(e)}")

            # 显示批量处理结果
            status.update(label=f"处理完成！共处理{len(new_files)}个文件", state="complete")
            st.session_state.show_final_toast = True

        # 显示最终汇总提示
        if st.session_state.show_final_toast:
            final_message = []
            if total_added['bunker'] + total_added['fuel'] > 0:
                final_message.append(f"• 新增油价记录: {total_added['bunker']}")
                final_message.append(f"• 新增燃料记录: {total_added['fuel']}")
            if error_messages:
                final_message.append(f"• 失败文件: {len(error_messages)}个")
            
            if final_message:
                st.toast("\n".join(["处理结果汇总:"] + final_message), icon="📊")
            
            st.session_state.show_final_toast = False

        # 清除缓存保证数据更新
        st.cache_data.clear()

    # 显示错误信息（独立显示）
    if error_messages:
        with st.container(border=True):
            st.error("\n\n".join(error_messages))

    # --------------------------
    # 数据分析模块
    # --------------------------
    bunker_df = load_history_data(bunker_path)
    fuel_df = load_history_data(fuel_path)

    with st.expander("📈 第二步 - 数据分析", expanded=True):
        tab1, tab2, tab3, tab4 = st.tabs(["港口油价信息", "油价趋势分析", "燃料价格分析", "数据对比"])

        with tab1:
            if not bunker_df.empty:
                st.subheader("近期油价趋势（最近10个记录）")
                recent_data = bunker_df.head(10).copy()  # 使用.copy()确保独立操作
                recent_data['Date'] = recent_data['Date'].astype(str)  # 转换日期列为字符串格式
                
                for region, ports in REGION_PORTS.items():
                    st.subheader(f"🏙️ {region}")
                    region_cols = [col for col in recent_data.columns if col in ports]
                    st.dataframe(
                        recent_data[["Date"] + region_cols].set_index("Date"),
                        use_container_width=True,
                        height=300,
                        hide_index=False
                    )
            else:
                st.warning("暂无油价数据可供分析。")

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
                filtered_df = bunker_df.loc[bunker_df['Date'].apply(lambda x: x.year) == selected_year]  # 使用.loc
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
            else:
                st.warning("暂无燃料价格数据。")

        with tab4:
            if not bunker_df.empty:
                st.subheader("指定日期港口价格对比")
                date_options = sorted(bunker_df['Date'].astype(str).unique(), reverse=True)
                col1, col2 = st.columns(2)
                with col1:
                    date1 = st.selectbox("选择对比日期1", date_options)
                with col2:
                    date2 = st.selectbox("选择对比日期2", date_options)
                if date1 and date2:
                    df1 = bunker_df.loc[bunker_df['Date'].astype(str) == date1]  # 使用.loc
                    df2 = bunker_df.loc[bunker_df['Date'].astype(str) == date2]  # 使用.loc
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

    # --------------------------
    # 数据下载模块
    # --------------------------
    with st.expander("📥 第三步 - 数据导出", expanded=True):
        st.subheader("完整数据下载")
        col1, col2 = st.columns(2)
        with col1:
            if bunker_path.exists():
                try:
                    data = generate_excel_download(bunker_df)
                    st.download_button(
                        label="下载完整油价数据",
                        data=data,
                        file_name="bunker_prices_full.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        on_click=lambda: on_download_click(True, "油价数据")
                    )
                except ValueError:
                    st.warning("油价数据为空，无法下载")
                    on_download_click(False, "")
        with col2:
            if fuel_path.exists():
                try:
                    data = generate_excel_download(fuel_df)
                    st.download_button(
                        label="下载完整燃料数据",
                        data=data,
                        file_name="fuel_prices_full.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        on_click=lambda: on_download_click(True, "燃料数据")
                    )
                except ValueError:
                    st.warning("燃料数据为空，无法下载")
                    on_download_click(False, "")

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
                    try:
                        data = generate_excel_download(daily_bunker)
                        st.download_button(
                            label="下载当日油价数据",
                            data=data,
                            file_name=f"bunker_{selected_bunker_date}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            on_click=lambda: on_download_click(True, f"当日油价数据 ({selected_bunker_date})")
                        )
                    except ValueError:
                        st.warning("当日油价数据为空，无法下载")
                        on_download_click(False, "")
        with col2:
            if not fuel_df.empty:
                selected_fuel_date = st.selectbox(
                    "选择燃料日期",
                    options=sorted(fuel_df['Date'].astype(str).unique(), reverse=True)
                )
                if selected_fuel_date:
                    daily_fuel = fuel_df[fuel_df['Date'].astype(str) == selected_fuel_date]
                    try:
                        data = generate_excel_download(daily_fuel)
                        st.download_button(
                            label="下载当日燃料数据",
                            data=data,
                            file_name=f"fuel_{selected_fuel_date}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            on_click=lambda: on_download_click(True, f"当日燃料数据 ({selected_fuel_date})")
                        )
                    except ValueError:
                        st.warning("当日燃料数据为空，无法下载")
                        on_download_click(False, "")

if __name__ == "__main__":
    main_ui()  

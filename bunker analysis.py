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
# 常量定义（新增部分）
# --------------------------
PORT_CODE_MAPPING = {
    # Asia Pacific/Middle East
    "MFSPD00": "Singapore", "MFFJD00": "Fujairah", "MFJPD00": "Japan",
    "BAMFB00": "West Japan", "MFSKD00": "South Korea", "WKMFA00": "South Korea (West)",
    "MFHKD00": "Hong Kong", "MFSHD00": "Shanghai", "MFZSD00": "Zhoushan",
    "MFDSY00": "Sydney", "MFDMB00": "Melbourne", "MFDKW00": "Kuwait",
    "MFDKF00": "Khor Fakkan", "MFDMM00": "Mumbai", "MFDCL00": "Colombo",
    # Europe
    "MFAGD00": "Algeciras", "MFDBD00": "Durban", "MFGBD00": "Gibraltar",
    "MFMLD00": "Malta", "MFPRD00": "Piraeus", "MFRDD00": "Rotterdam",
    "MFDAN00": "Antwerp", "MFDGT00": "Gothenburg", "MFDHB00": "Hamburg",
    "MFDIS00": "Istanbul", "MFDLP00": "Las Palmas", "MFDNV00": "Novorossiisk",
    "MFDPT00": "St. Petersburg", "MFLIS00": "Lisbon", "MFLOM00": "Lome",
    # Americas
    "MFHOD00": "Houston", "MFNYD00": "New York", "MFLAD00": "Los Angeles",
    "MFNOD00": "New Orleans", "MFPAD00": "Philadelphia", "MFSED00": "Seattle",
    "MFVAD00": "Vancouver", "MFBAD00": "Buenos Aires", "MFCRD00": "Cartagena",
    "MFSAD00": "Santos", "AMFVA00": "Valparaiso", "AMFCA00": "Callao",
    "AMFGY00": "Guayaquil", "AMFLB00": "La Libertad", "AMFMT00": "Montevideo",
    "AMFSF00": "San Francisco", "AMFMO00": "Montreal*"
}

REGION_CODE_ORDER = [
    ("Asia Pacific/Middle East", [
        "MFSPD00", "MFFJD00", "MFJPD00", "BAMFB00", "MFSKD00",
        "WKMFA00", "MFHKD00", "MFSHD00", "MFZSD00", "MFDSY00",
        "MFDMB00", "MFDKW00", "MFDKF00", "MFDMM00", "MFDCL00"
    ]),
    ("Europe", [
        "MFAGD00", "MFDBD00", "MFGBD00", "MFMLD00", "MFPRD00",
        "MFRDD00", "MFDAN00", "MFDGT00", "MFDHB00", "MFDIS00",
        "MFDLP00", "MFDNV00", "MFDPT00", "MFLIS00", "MFLOM00"
    ]),
    ("Americas", [
        "MFHOD00", "MFNYD00", "MFLAD00", "MFNOD00", "MFPAD00",
        "MFSED00", "MFVAD00", "MFBAD00", "MFCRD00", "MFSAD00",
        "AMFVA00", "AMFCA00", "AMFGY00", "AMFLB00", "AMFMT00",
        "AMFSF00", "AMFMO00"
    ])
]

TAB1_COLUMN_ORDER = [
    "AMFSA00", "MFSPD00", "PPXDK00", "PUAFT00", "AAXYO00",
    "PUMFD00", "MFRDD00", "PUABC00", "PUAFN00", "AARTG00",
    "MFSAD00", "AAXWO00", "MFZSD00", "BFDZA00", "MGZSD00",
    "MFHKD00", "PUAER00", "AAXYQ00", "MFSKD00", "PUAGQ00",
    "AAXYS00", "MFSHD00", "AARKD00", "AAXYR00", "MFGBD00",
    "AAKAB00", "AARSU00", "MFNOD00", "AAGQE00", "AAWYA00",
    "AMFFA00", "MFFJD00", "PUAXP00", "AAXYP00"
]

FUEL_TYPE_MAPPING = {
    "MLBSO00": "MEOH-SG",
    "LNBSF00": "LNG-SG"
}

# --------------------------
# 数据处理工具类（修改部分）
# --------------------------
class BunkerDataProcessor:
    @staticmethod
    def format_date(date_series: pd.Series) -> pd.Series:
        # 增强日期处理逻辑
        return pd.to_datetime(
            date_series,
            errors='coerce',  # 将无效日期转为NaT
            format='mixed'    # 自动识别多种日期格式
        ).dt.date

    @staticmethod
    def clean_dataframe(df: pd.DataFrame, data_type: str = "bunker") -> pd.DataFrame:
        if df.empty:
            return df

        # 统一日期格式（增强处理）
        if 'Date' in df.columns:
            # 先转换为datetime类型
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # 过滤无效日期
            df = df[df['Date'].notna()]
            # 转换为date类型
            df['Date'] = df['Date'].dt.date

        # 按数据类型处理列顺序
        if data_type == "bunker":
            all_ports = [PORT_CODE_MAPPING.get(code, code) for code in TAB1_COLUMN_ORDER if code in df.columns]
            ordered_columns = ['Date'] + all_ports
            df = df.reindex(columns=ordered_columns, fill_value=pd.NA)
        elif data_type == "regional":
            ordered_columns = ['Date']
            for _, codes in REGION_CODE_ORDER:
                ordered_columns.extend([PORT_CODE_MAPPING[code] for code in codes if code in df.columns])
            df = df.reindex(columns=ordered_columns, fill_value=pd.NA)
        elif data_type == "fuel":
            ordered_columns = ['Date'] + list(FUEL_TYPE_MAPPING.values())
            df = df.rename(columns=FUEL_TYPE_MAPPING)
            df = df.reindex(columns=ordered_columns, fill_value=pd.NA)

        # 去重排序
        df = df.drop_duplicates()
        df = df.sort_values('Date', ascending=True)
        return df.reset_index(drop=True)

    @staticmethod
    def merge_data(existing_df: pd.DataFrame, new_df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        combined = pd.concat([existing_df, new_df])
        return BunkerDataProcessor.clean_dataframe(combined, data_type)

# --------------------------
# GitHub数据管理器（修改部分）
# --------------------------
class GitHubDataManager:
    def __init__(self, token: str, repo_name: str):
        self.token = token
        self.repo_name = repo_name
        self.g = Github(self.token)
        self.repo = self.g.get_repo(self.repo_name)

    def read_excel(self, file_path: str, data_type: str) -> Tuple[pd.DataFrame, bool]:
        try:
            contents = self.repo.get_contents(file_path)
            df = pd.read_excel(BytesIO(base64.b64decode(contents.content)), sheet_name=0)
            return BunkerDataProcessor.clean_dataframe(df, data_type), True
        except Exception as e:
            return pd.DataFrame(), False

    def save_excel(self, df: pd.DataFrame, file_path: str, commit_msg: str, data_type: str) -> bool:
        try:
            # 确保路径存在
            dir_path = os.path.dirname(file_path)
            if dir_path:
                try:
                    # 检查目录是否存在
                    self.repo.get_contents(dir_path)
                except Exception as e:
                    # 创建目录
                    self.repo.create_file(
                        path=f"{dir_path}/.gitkeep",
                        message=f"Create directory {dir_path}",
                        content=""
                    )
            # 按数据类型处理列顺序
            processed_df = BunkerDataProcessor.clean_dataframe(df, data_type)
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                processed_df.to_excel(writer, index=False)
            content = output.getvalue()
            
            try:
                contents = self.repo.get_contents(file_path)
                self.repo.update_file(contents.path, commit_msg, content, contents.sha)
            except:
                self.repo.create_file(file_path, commit_msg, base64.b64encode(content).decode())
            return True
        except Exception as e:
            logger.error(f"GitHub保存失败: {str(e)}")
            return False

# --------------------------
# PDF数据提取器（修改后的版本，结合代码1的区域划分逻辑）
# --------------------------
class EnhancedBunkerPriceExtractor:
    def __init__(self, pdf_path: str, bunker_path: str, fuel_path: str):
        self.pdf_path = pdf_path
        self.bunker_path = bunker_path
        self.fuel_path = fuel_path

    def process_pdf(self) -> Dict[str, int]:
        try:
            doc = fitz.open(self.pdf_path)
            result = {'bunker': 0, 'fuel': 0}
            
            # 处理第一页（油价数据）
            bunker_df = self._process_bunker_page(doc[0])
            if bunker_df is not None:
                success = self._save_data(bunker_df, self.bunker_path, "bunker")
                result['bunker'] = len(bunker_df) if success else 0

            # 处理第二页（燃料数据）
            fuel_df = self._process_fuel_page(doc[1])
            if fuel_df is not None:
                success = self._save_data(fuel_df, self.fuel_path, "fuel")
                result['fuel'] = len(fuel_df) if success else 0

            doc.close()
            return result
        except Exception as e:
            logger.error(f"PDF处理失败: {str(e)}")
            return {'bunker': 0, 'fuel': 0}

    def _get_page_coordinates(self, page, config: Dict) -> Optional[Dict]:
        """获取页面关键区域坐标"""
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

    def _extract_text_from_area(self, page, coords: Dict) -> str:
        """从指定区域提取文本"""
        rect = fitz.Rect(
            coords['left_x'],
            min(coords['start_y'], coords['end_y']),
            coords['right_x'],
            max(coords['start_y'], coords['end_y'])
        )
        return page.get_text("text", clip=rect)

    def _process_bunker_page(self, page) -> Optional[pd.DataFrame]:
        # 定义页面解析配置
        coord_config = {
            'start_key': 'Bunkerwire',
            'end_key': 'Ex-Wharf',
            'right_boundary': 'Marine Fuel (PGB page 30)'
        }
        coords = self._get_page_coordinates(page, coord_config)
        if not coords:
            return None

        # 提取指定区域的文本
        raw_text = self._extract_text_from_area(page, coords)
        if not raw_text:
            return None

        date = self._extract_date(raw_text)
        if not date:
            return None

        # 使用精确正则表达式匹配
        pattern = re.compile(r"([A-Z]{6,8}00)\s+(\d+\.\d+|NA)\s+([+-]?\d+\.\d+|NANA)")
        matches = pattern.findall(raw_text)
        if not matches:
            return None

        # 构建数据字典
        data = {'Date': [date]}
        for code, price, _ in matches:
            if price != 'NA' and code in PORT_CODE_MAPPING:
                port = PORT_CODE_MAPPING[code]
                data[port] = [float(price)]

        return pd.DataFrame(data)

    def _process_fuel_page(self, page) -> Optional[pd.DataFrame]:
        # 定义页面解析配置
        coord_config = {
            'start_key': 'Alternative marine fuels',
            'end_key': 'Arab Gulf'
        }
        coords = self._get_page_coordinates(page, coord_config)
        if not coords:
            return None

        # 提取指定区域的文本
        raw_text = self._extract_text_from_area(page, coords)
        if not raw_text:
            return None

        date = self._extract_date(raw_text)
        if not date:
            return None

        pattern = re.compile(r"(MLBSO00|LNBSF00)\s+(\d+\.\d+|NA)")
        matches = pattern.findall(raw_text)
        if not matches:
            return None

        data = {'Date': [date]}
        for code, price in matches:
            if price != 'NA':
                data[code] = [float(price)]
        return pd.DataFrame(data)

    def _extract_date(self, text: str) -> Optional[datetime.date]:
        date_pattern = r"Volume\s+\d+\s+/\s+Issue\s+\d+\s+/\s+(\w+\s+\d{1,2},\s+\d{4})"
        match = re.search(date_pattern, text)
        if match:
            try:
                return datetime.strptime(match.group(1), "%B %d, %Y").date()
            except ValueError:
                pass
        return None

    def _save_data(self, new_df: pd.DataFrame, output_path: str, data_type: str) -> bool:
        try:
            github_token = st.secrets.github.token
            repo_name = st.secrets.github.repo
            gh_manager = GitHubDataManager(github_token, repo_name)
            
            existing_df, exists = gh_manager.read_excel(output_path, data_type)
            if exists:
                combined_df = BunkerDataProcessor.merge_data(existing_df, new_df, data_type)
            else:
                combined_df = BunkerDataProcessor.clean_dataframe(new_df, data_type)
                
            return gh_manager.save_excel(combined_df, output_path, 
                                      f"Update {data_type} data", 
                                      data_type)
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
            return False

# --------------------------
# Streamlit界面
# --------------------------
@st.cache_data(ttl=3600, show_spinner="加载历史数据...")
def load_history_data(path: str, data_type: str) -> pd.DataFrame:
    try:
        github_token = st.secrets.github.token
        repo_name = st.secrets.github.repo
        gh_manager = GitHubDataManager(github_token, repo_name)
        df, exists = gh_manager.read_excel(path, data_type)
        return df if exists else pd.DataFrame()
    except Exception as e:
        logger.error(f"数据加载失败: {path} - {str(e)}")
        return pd.DataFrame()

def generate_excel_download(df: pd.DataFrame) -> bytes:
    """增强的空数据校验"""
    if df.empty:
        raise ValueError("无法导出空数据集")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.getvalue()

def on_download_click(success: bool, filename: str):
    """下载状态提示"""
    if success:
        st.toast(f"✅ {filename} 开始下载，请查看浏览器下载列表")
    else:
        st.toast(f"⚠️ 下载文件生成失败", icon="⚠️")

def main_ui():
    st.set_page_config(page_title="船燃价格分析系统", layout="wide")
    st.title("Mariners' Bunker Price Analysis System")

    # 初始化路径
    BUNKER_PATH = "data/bunker_prices.xlsx"
    FUEL_PATH = "data/fuel_prices.xlsx"

    # 初始化session状态
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

    # 文件上传模块
    with st.expander("📤 第一步 - 上传PDF报告", expanded=True):
        uploaded_files = st.file_uploader(
            "选择Bunkerwire PDF报告（支持多选）",
            type=["pdf"],
            accept_multiple_files=True
        )

    # 处理新文件
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if new_files:
        with st.status("正在解析文件...", expanded=True) as status:
            progress_bar = st.progress(0)
            total_files = len(new_files)
            total_added = {'bunker': 0, 'fuel': 0}
            
            for idx, file in enumerate(new_files):
                try:
                    # 显示处理进度
                    progress_bar.progress((idx+1)/total_files, text=f"正在处理 {file.name} ({idx+1}/{total_files})")
                    
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(file.getbuffer())
                        extractor = EnhancedBunkerPriceExtractor(tmp.name, BUNKER_PATH, FUEL_PATH)
                        result = extractor.process_pdf()
                        total_added['bunker'] += result['bunker']
                        total_added['fuel'] += result['fuel']
                        st.session_state.processed_files.add(file.name)
                    os.unlink(tmp.name)
                    
                    # 实时显示处理结果
                    st.write(f"✅ {file.name} 处理完成 (油价+{result['bunker']}, 燃料+{result['fuel']})")
                except Exception as e:
                    st.error(f"处理失败 {file.name}: {str(e)}")
            
            status.update(label=f"处理完成！共新增油价{total_added['bunker']}条，燃料{total_added['fuel']}条", state="complete")
            st.cache_data.clear()

    # 数据分析模块
    bunker_df = load_history_data(BUNKER_PATH, "bunker")
    fuel_df = load_history_data(FUEL_PATH, "fuel")

    with st.expander("📈 第二步 - 数据分析", expanded=True):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "指定格式油价", "区域油价", "燃料价格", "趋势分析", "数据对比"
        ])

        # Tab1 - 指定格式油价
        with tab1:
            if not bunker_df.empty:
                st.subheader("标准格式油价数据")
                
                # 生成显示用数据
                display_cols = ['Date'] + [
                    PORT_CODE_MAPPING.get(code, code) 
                    for code in TAB1_COLUMN_ORDER 
                    if PORT_CODE_MAPPING.get(code, code) in bunker_df.columns
                ]
                display_df = bunker_df[display_cols].sort_values('Date', ascending=False)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=600
                )
                
                # 下载按钮
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "下载完整数据",
                        data=generate_excel_download(bunker_df[display_cols].sort_values('Date')),
                        file_name="standard_bunker.xlsx"
                    )
                with col2:
                    selected_date = st.selectbox("选择日期", bunker_df['Date'].unique())
                    st.download_button(
                        "下载单日数据",
                        data=generate_excel_download(
                            bunker_df[bunker_df['Date'] == selected_date][display_cols]),
                        file_name=f"standard_bunker_{selected_date}.xlsx"
                    )

        # Tab2 - 区域油价
        with tab2:
            if not bunker_df.empty:
                st.subheader("区域油价数据")
                
                # 生成区域数据
                for region_name, codes in REGION_CODE_ORDER:
                    port_names = [PORT_CODE_MAPPING[code] for code in codes if code in bunker_df.columns]
                    if port_names:
                        with st.expander(region_name, expanded=True):
                            region_df = bunker_df[['Date'] + port_names].sort_values('Date', ascending=False)
                            st.dataframe(region_df, height=300)
                
                # 下载按钮
                full_cols = ['Date']
                for _, codes in REGION_CODE_ORDER:
                    full_cols.extend([PORT_CODE_MAPPING[code] for code in codes if code in bunker_df.columns])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "下载完整区域数据",
                        data=generate_excel_download(bunker_df[full_cols].sort_values('Date')),
                        file_name="regional_bunker.xlsx"
                    )
                with col2:
                    selected_date = st.selectbox("选择日期", bunker_df['Date'].unique(), key="region_date")
                    st.download_button(
                        "下载单日区域数据",
                        data=generate_excel_download(
                            bunker_df[bunker_df['Date'] == selected_date][full_cols]),
                        file_name=f"regional_bunker_{selected_date}.xlsx"
                    )

        with tab3:
            if not fuel_df.empty:
                st.subheader("替代燃料价格趋势")
                
                # 转换列名为友好名称
                fuel_display_df = fuel_df.rename(columns=FUEL_TYPE_MAPPING)
                
                # 显示最新10条数据（最新日期在上）
                st.dataframe(
                    fuel_display_df.sort_values('Date', ascending=False).head(10),
                    use_container_width=True,
                    height=300
                )
                
                # 趋势图
                fig = go.Figure()
                for fuel_type in FUEL_TYPE_MAPPING.values():
                    if fuel_type in fuel_display_df.columns:
                        fig.add_trace(go.Scatter(
                            x=fuel_display_df['Date'],
                            y=fuel_display_df[fuel_type],
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
                
                # 下载逻辑
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="下载完整燃料数据",
                        data=generate_excel_download(fuel_display_df.sort_values('Date')),
                        file_name="fuel_prices_full.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col2:
                    selected_date = st.selectbox(
                        "选择燃料日期",
                        options=sorted(fuel_df['Date'].astype(str).unique(), reverse=True),
                        key="fuel_date"
                    )
                    st.download_button(
                        label="下载单日燃料数据",
                        data=generate_excel_download(
                            fuel_display_df[fuel_display_df['Date'].astype(str) == selected_date]
                        ),
                        file_name=f"fuel_{selected_date}.xlsx"
                    )
            else:
                st.warning("暂无燃料数据")

        # Tab4 - 趋势分析（原TAB3）
        with tab4:
            if not bunker_df.empty:
                # 添加类型转换确保Date列是datetime类型
                bunker_df['Date'] = pd.to_datetime(bunker_df['Date'], errors='coerce')
                bunker_df = bunker_df.dropna(subset=['Date'])
                
                st.subheader("多港口趋势对比分析")
                
                # 控件布局
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    # 按区域分组选择港口
                    selected_ports = []
                    for region_name, codes in REGION_CODE_ORDER:
                        port_names = [PORT_CODE_MAPPING[code] for code in codes 
                                    if PORT_CODE_MAPPING[code] in bunker_df.columns]
                        if port_names:
                            selected = st.multiselect(
                                f"{region_name} 港口",
                                options=port_names,
                                default=port_names[:1] if region_name == "Asia Pacific/Middle East" else []
                            )
                            selected_ports.extend(selected)
                with col2:
                    # 修改年份获取方式
                    if not bunker_df.empty:
                        bunker_df['Year'] = bunker_df['Date'].dt.year  # 现在可以安全使用.dt
                        year_options = sorted(bunker_df['Year'].unique(), reverse=True)
                        selected_year = st.selectbox("选择年份", year_options)
                with col3:
                    st.markdown("###")
                    show_annotations = st.checkbox("显示数据点", value=True)
                
                # 数据处理
                filtered_df = bunker_df[bunker_df['Year'] == selected_year]
                if selected_ports:
                    # 创建趋势图
                    fig = go.Figure()
                    for port in selected_ports:
                        valid_data = filtered_df[['Date', port]].dropna()
                        if not valid_data.empty:
                            fig.add_trace(go.Scatter(
                                x=valid_data['Date'],
                                y=valid_data[port],
                                name=port,
                                mode='lines+markers' if show_annotations else 'lines',
                                visible=True
                            ))
                    
                    # 图表布局
                    fig.update_layout(
                        title=f"{selected_year}年油价趋势分析",
                        xaxis_title="日期",
                        yaxis_title="价格 (USD)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        hovermode="x unified",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("请至少选择一个港口进行对比")
            else:
                st.warning("暂无油价数据可供分析")

        # Tab5 - 数据对比（原TAB4）
        with tab5:
            if not bunker_df.empty:
                st.subheader("跨日期数据对比分析")
                # 确保日期列是datetime类型
                bunker_df['Date'] = pd.to_datetime(bunker_df['Date'])
                
                # 日期选择
                date_options = bunker_df['Date'].dt.strftime('%Y-%m-%d').sort_values(ascending=False).unique()
                col1, col2 = st.columns(2)
                with col1:
                    date1 = st.selectbox("对比日期1", date_options, index=0)
                with col2:
                    date2 = st.selectbox("对比日期2", date_options, index=1 if len(date_options)>1 else 0)
                
                # 数据获取
                df1 = bunker_df[bunker_df['Date'].astype(str) == date1]
                df2 = bunker_df[bunker_df['Date'].astype(str) == date2]
                
                if not df1.empty and not df2.empty:
                    # 按区域生成对比数据
                    comparison_data = []
                    for region_name, codes in REGION_CODE_ORDER:
                        region_ports = [PORT_CODE_MAPPING[code] for code in codes 
                                       if PORT_CODE_MAPPING[code] in bunker_df.columns]
                        for port in region_ports:
                            price1 = df1[port].values[0] if port in df1.columns else None
                            price2 = df2[port].values[0] if port in df2.columns else None
                            if price1 is not None and price2 is not None:
                                change = ((price1 - price2) / price2 * 100) if price2 != 0 else 0
                                comparison_data.append({
                                    "Region": region_name,
                                    "Port": port,
                                    date1: price1,
                                    date2: price2,
                                    "变化率 (%)": f"{change:.2f}%"
                                })
                    
                    # 显示对比表格
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # 按区域分组显示
                        for region in REGION_CODE_ORDER:
                            region_name = region[0]
                            region_df = comparison_df[comparison_df['Region'] == region_name]
                            if not region_df.empty:
                                with st.expander(f"{region_name} 对比数据", expanded=True):
                                    st.dataframe(
                                        region_df.drop('Region', axis=1).set_index('Port'),
                                        use_container_width=True,
                                        height=300
                                    )
                    else:
                        st.warning("选定日期无有效数据对比")
                else:
                    st.warning("未找到选定日期的数据")
            else:
                st.warning("暂无数据可供对比")

if __name__ == "__main__":
    main_ui()

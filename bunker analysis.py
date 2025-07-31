import fitz
import re
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import streamlit as st
import tempfile
from io import BytesIO
from github import Github
import base64
import requests

# 配置日志
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# 定义按页码分组的正则表达式
EXTRACTION_CONFIG = {
    1: [  
        r"(MFSPD00)\s+(\d+\.\d+)",  
        r"(MFRDD00)\s+(\d+\.\d+)",  
        r"(MFHKD00)\s+(\d+\.\d+)",  
        r"(MFSAD00)\s+(\d+\.\d+)",  
        r"(MFZSD00)\s+\s*(\d+\.\d+)",  
    ],
    2: [  
        r"(MLBSO00)\s+(\d+\.\d+)",  
        r"(LNBSF00)\s+(\d+\.\d+)",  
    ]
}
DATE_PATTERN = r"Volume\s+\d+\s+/\s+Issue\s+\d+\s+/\s+(\w+\s+\d{1,2},\s+\d{4})"

class PDFDataExtractor:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def extract_data(self) -> pd.DataFrame:
        doc = fitz.open(self.pdf_path)  
        extracted_data = {}  
        date = None  

        for page_num in range(len(doc)):  
            page = doc.load_page(page_num)  
            text = page.get_text().strip()  

            if page_num == 0:  
                date_match = re.search(DATE_PATTERN, text)
                if date_match:
                    date_str = date_match.group(1)
                    date = datetime.strptime(date_str, "%B %d, %Y").date()
                    extracted_data["Date"] = date

            patterns = EXTRACTION_CONFIG.get(page_num + 1, [])  
            for pattern in patterns:  
                matches = re.findall(pattern, text)
                for match in matches:  
                    code, value = match
                    extracted_data[code] = float(value)

        doc.close()  

        desired_order = [
            "Date",
            "MFSPD00", "MFRDD00", "MFHKD00", "MFSAD00", "MFZSD00", "MLBSO00", "LNBSF00"
        ]
        sorted_data = {key: extracted_data.get(key) for key in desired_order}  
        return pd.DataFrame([sorted_data])  

class GitHubDataSaver:
    def __init__(self, repo_name: str, file_path: str, github_token: str):
        self.repo_name = repo_name
        self.file_path = file_path
        self.github_token = github_token

    def save_data(self, df: pd.DataFrame):
        if df.empty:
            return

        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        current_date = df['Date'].iloc[0]

        try:
            # 连接到GitHub
            g = Github(self.github_token)
            repo = g.get_repo(self.repo_name)
            
            # 尝试获取现有文件
            try:
                contents = repo.get_contents(self.file_path)
                existing_data = pd.read_excel(BytesIO(base64.b64decode(contents.content)), engine='openpyxl')
                if current_date in existing_data['Date'].tolist():
                    st.warning(f"数据日期 {current_date} 已存在，跳过保存。")
                    return
                combined_df = pd.concat([existing_data, df])
            except Exception as e:
                # 文件不存在时创建新文件
                combined_df = df

            # 处理数据
            combined_df = combined_df.sort_values(by='Date', ascending=True)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                combined_df.to_excel(writer, index=False)
            excel_data = output.getvalue()

            # 提交到GitHub
            if 'contents' in locals():
                repo.update_file(contents.path, f"更新数据 {datetime.today().date()}", excel_data, contents.sha)
            else:
                repo.create_file(self.file_path, "初始数据提交", excel_data)
            
            st.success("数据已保存到GitHub仓库！")
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
            st.error(f"保存数据时出错: {str(e)}")

def main_ui():
    st.set_page_config(page_title="PDF 数据提取器", layout="wide")
    st.title("PDF 数据提取器")

    # 从Secrets获取GitHub配置
    try:
        github_token = st.secrets.github.token
        repo_name = st.secrets.github.repo
        file_path = "history_data/extracted_data.xlsx"
    except Exception as e:
        st.error("请正确配置GitHub Secrets！")
        return

    # 文件上传模块
    with st.expander("📤 第一步 - 上传PDF文件", expanded=True):
        uploaded_file = st.file_uploader("选择PDF文件", type=["pdf"])

    if uploaded_file:
        current_file_hash = hash(uploaded_file.getvalue())
        if 'last_file_hash' not in st.session_state or st.session_state.last_file_hash != current_file_hash:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    pdf_path = Path(tmp.name)

                extractor = PDFDataExtractor(pdf_path)
                extracted_df = extractor.extract_data()

                saver = GitHubDataSaver(repo_name, file_path, github_token)
                saver.save_data(extracted_df)

                st.session_state.last_file_hash = current_file_hash
                st.success("数据提取成功！")
            except Exception as e:
                logger.error(f"处理失败: {str(e)}")
                st.error(f"文件处理错误: {str(e)}")

    # 数据展示模块
    with st.expander("📈 第二步 - 数据展示", expanded=True):
        try:
            # 从GitHub获取数据
            g = Github(github_token)
            repo = g.get_repo(repo_name)
            contents = repo.get_contents(file_path)
            data_df = pd.read_excel(BytesIO(base64.b64decode(contents.content)), engine='openpyxl')
            data_df['Date'] = pd.to_datetime(data_df['Date']).dt.strftime('%Y-%m-%d')
            data_df = data_df.sort_values(by='Date', ascending=False)
            
            # 展示内容一：选定两个日期进行比较
            st.subheader("日期数据变动比较")
            date_options = data_df['Date'].unique()
            selected_dates = st.multiselect("选择两个日期进行比较", options=date_options, max_selections=2)
            if len(selected_dates) == 2:
                date1, date2 = selected_dates
                data1 = data_df[data_df['Date'] == date1][["MFSPD00", "MFRDD00", "MFHKD00", "MFSAD00", "MFZSD00"]]
                data2 = data_df[data_df['Date'] == date2][["MFSPD00", "MFRDD00", "MFHKD00", "MFSAD00", "MFZSD00"]]
                comparison_df = pd.DataFrame({
                    "指标": ["MFSPD00", "MFRDD00", "MFHKD00", "MFSAD00", "MFZSD00"],
                    f"{date1}": data1.iloc[0].values,
                    f"{date2}": data2.iloc[0].values,
                    "变动": data1.iloc[0].values - data2.iloc[0].values
                })
                st.table(comparison_df.set_index("指标"))

            # 展示内容二：展示"MLBSO00", "LNBSF00"的最新十个日期数据
            st.subheader("最新十个日期的MLBSO00和LNBSF00数据")
            latest_data = data_df[["Date", "MLBSO00", "LNBSF00"]].head(10)
            st.table(latest_data.set_index("Date"))
        except Exception as e:
            st.warning("暂无历史数据或读取失败")

    # 数据导出模块（保持原样，但数据源改为GitHub）
    with st.expander("📥 第三步 - 数据导出", expanded=True):
        try:
            contents = repo.get_contents(file_path)
            data_df = pd.read_excel(BytesIO(base64.b64decode(contents.content)), engine='openpyxl')
            
            # 新增日期处理逻辑
            data_df['Date'] = pd.to_datetime(data_df['Date'])  # 转换为datetime类型
            data_df = data_df.sort_values('Date', ascending=False)  # 按日期降序排列
            
            # 生成排序后的日期选项
            sorted_dates = data_df['Date'].dt.strftime('%Y-%m-%d').unique()
            
            st.subheader("按日期下载")
            selected_date = st.selectbox("选择日期", options=sorted_dates)  # 显示排序后的日期
            
            daily_data = data_df[data_df['Date'] == pd.to_datetime(selected_date)]
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                daily_data.to_excel(writer, index=False)
            st.download_button(
                label="下载选定日期数据",
                data=output.getvalue(),
                file_name=f"data_{selected_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.warning("暂无数据可供导出")

if __name__ == "__main__":
    main_ui()

import streamlit as st
import os
import re
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.platypus import Table, TableStyle, Image
from reportlab.lib.utils import ImageReader
from io import BytesIO
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_JUSTIFY
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import timedelta
from matplotlib import ticker
from datetime import datetime

# Cấu hình giao diện
st.set_page_config(page_title="ĐỒ ÁN CUỐI KỲ", page_icon="📄", layout="centered")

# Chèn CSS để đổi font chữ Streamlit
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        background-color: #121212;
        color: #FFFFFF;
    }

    h1 {
        font-size: 40px !important;
        font-weight: 700 !important;
        color: #FFFFFF;
        text-align: center;
    }

    h2 {
        font-size: 32px !important;
        font-weight: 600 !important;
        color: #E0E0E0;
        text-align: center;
    }

    p, label, div {
        font-size: 18px !important;
        font-weight: 500 !important;
        color: #F5F5F5;
    }

    .stSelectbox, .stTextInput {
        background-color: #1E1E1E;
        color: #FFFFFF;
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Màu chủ đạo
LIGHT_GREEN = colors.HexColor("#8ed1a1")
LIGHT_GREEN_BG = HexColor("#ccffda")
GREEN_TEXT = colors.HexColor("#2e7d32")

# Load data
DATA_PATH = "C:/Users/Admin/Downloads/data"
IMG_PATH = os.path.join(DATA_PATH, "arrow_down_red.png")
IMG_PATH1 = os.path.join(DATA_PATH, "arrow_up_green.png")
LOGO_PATH = os.path.join(DATA_PATH, "logo.png")

def load_data():
    df_basic = pd.read_csv(os.path.join(DATA_PATH, "tm.csv"))
    df_info = pd.read_excel(os.path.join(DATA_PATH, "Info.xlsx"))
    df_price = pd.read_csv(os.path.join(DATA_PATH, "Price.csv"), dtype={"Code": str}, low_memory=False)
    df_price.set_index("Code", inplace=True)
    df_price = df_price.T
    df_price.index = pd.to_datetime(df_price.index)
    df_ratio = pd.read_excel(os.path.join(DATA_PATH, "ratio.xlsx"))
    bcdkt_df = pd.read_csv(os.path.join(DATA_PATH, "BCDKT.csv"))
    kqkd_df = pd.read_csv(os.path.join(DATA_PATH, "KQKD.csv"))
    lctt_df = pd.read_csv(os.path.join(DATA_PATH, "LCTT.csv"))
    # Load Market Cap
    marketcap_df = pd.read_excel(os.path.join(DATA_PATH, "Cleaned_Vietnam_Marketcap.xlsx"), sheet_name="Sheet2")
    marketcap_df.rename(columns={"Mã": "Code"}, inplace=True)
    marketcap_df.set_index("Code", inplace=True)
    marketcap_df = marketcap_df.drop(columns=["Name"], errors="ignore")
    marketcap_df = marketcap_df.T
    marketcap_df.index = pd.to_datetime(marketcap_df.index)

    return df_basic, df_info, df_price, df_ratio, bcdkt_df, kqkd_df, lctt_df, marketcap_df

df, info_df, price_df, ratio_df, bcdkt_df, kqkd_df, lctt_df, marketcap_df = load_data()

# Lấy danh sách ngày hợp lệ từ file Price.csv
min_date, max_date = price_df.index.min(), price_df.index.max()

# Đăng ký font hỗ trợ tiếng Việt
pdfmetrics.registerFont(TTFont("Roboto_Black", os.path.join(DATA_PATH, "Roboto_Condensed-Black.ttf")))
pdfmetrics.registerFont(TTFont("Roboto_Regular", os.path.join(DATA_PATH, "Roboto_SemiCondensed-Regular.ttf")))

# Hàm vẽ biểu đồ giá
def draw_marketcap_chart(marketcap_df, stock_code):
    stock_code_mv = f"{stock_code}(MV)"
    if stock_code_mv not in marketcap_df.columns:
        stock_code_mv = stock_code  # fallback

    data = marketcap_df[stock_code_mv]
    data = data[data.index <= pd.to_datetime("2024-12-31")]
    data = data / 1e3  # đổi sang tỷ đồng

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(data.index, data.values, color="green", linewidth=1.5)
    ax.set_title(f"Giá trị vốn hoá thị trường của {stock_code}", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_stock_price(stock_code):
    if stock_code not in price_df.columns:
        print(f"Mã {stock_code} không tồn tại trong dữ liệu.")
        return None

    stock_price = price_df[stock_code].dropna()

    if stock_price.empty:
        print(f"Không có dữ liệu đủ để vẽ biểu đồ cho {stock_code}.")
        return None

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_price.index, stock_price.values, linestyle='-', color='green')
    ax.grid(True)

    # Định dạng trục y với dấu phẩy
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Hiển thị năm trên trục X
    years = stock_price.index.year.unique()
    ax.set_xticks([pd.Timestamp(year=year, month=1, day=1) for year in years])
    ax.set_xticklabels(years)
    ax.set_aspect('auto')

    # Chỉnh trục y sang bên phải
    ax.yaxis.set_label_position('right')  # Đổi vị trí nhãn trục y sang bên phải
    ax.yaxis.tick_right()  # Đưa các dấu tick của trục y sang bên phải

    # Tạo trục y thứ hai ở bên phải (optional)
    ax2 = ax.twinx()  # Trục y phụ ở bên phải
    ax2.set_ylim(ax.get_ylim())  # Giới hạn trục y phụ giống như trục y chính
    # Định dạng trục y với dấu phẩy
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Lưu biểu đồ vào buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return buffer

def plot_stock_price1(stock_code, selected_date):
    if stock_code not in price_df.columns:
        print(f"Mã {stock_code} không tồn tại trong dữ liệu.")
        return None

    stock_price = price_df[stock_code].dropna()

    if stock_price.empty:
        print(f"Không có dữ liệu đủ để vẽ biểu đồ cho {stock_code}.")
        return None

    # Chuyển selected_date thành datetime
    selected_date = pd.to_datetime(selected_date)

    # Lọc dữ liệu trong vòng 6 tháng
    start_date = selected_date - timedelta(days=180)  # 180 ngày là 6 tháng
    stock_price = stock_price[(stock_price.index >= start_date) & (stock_price.index <= selected_date)]

    if stock_price.empty:
        print(f"Không có dữ liệu trong 6 tháng trước {selected_date}.")
        return None

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_price.index, stock_price.values, linestyle='-', color='green')
    ax.grid(True)

    # Định dạng trục y với dấu phẩy
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Hiển thị theo tháng/năm trên trục X
    ax.set_xticks(stock_price.index[::30])  # Chọn một điểm mỗi 30 ngày để hiển thị trên trục X
    ax.set_xticklabels([date.strftime('%m/%Y') for date in stock_price.index[::30]])  # Hiển thị theo định dạng tháng/năm
    ax.set_aspect('auto')

    # Chỉnh trục y sang bên phải
    ax.yaxis.set_label_position('right')  # Đổi vị trí nhãn trục y sang bên phải
    ax.yaxis.tick_right()  # Đưa các dấu tick của trục y sang bên phải

    # Tạo trục y thứ hai ở bên phải (optional)
    ax2 = ax.twinx()  # Trục y phụ ở bên phải
    ax2.set_ylim(ax.get_ylim())  # Giới hạn trục y phụ giống như trục y chính
    # Định dạng trục y với dấu phẩy
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Lưu biểu đồ vào buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return buffer

def calculate_percentage_change(price_df, selected_date, stock_code):
    date_ranges = {
        "1 ngày": 1,
        "5 ngày": 5,
        "3 tháng": 90,
        "6 tháng": 180,
        "1 năm": 365
    }

    # Đảm bảo kiểu datetime
    selected_date = pd.to_datetime(selected_date).normalize()
    price_df.index = pd.to_datetime(price_df.index).normalize()

    percentage_changes = {}

    # Kiểm tra xem stock_code có tồn tại không
    if stock_code not in price_df.columns:
        return {label: "Không có dữ liệu" for label in date_ranges}

    # Kiểm tra ngày hiện tại có giá không
    if selected_date not in price_df.index or pd.isna(price_df.loc[selected_date, stock_code]):
        return {label: "Không có dữ liệu" for label in date_ranges}

    current_price = price_df.loc[selected_date, stock_code]

    for label, num_days in date_ranges.items():
        # Tìm ngày gần nhất trước đó
        past_target_date = selected_date - pd.Timedelta(days=num_days)
        past_dates = price_df.index[(price_df.index <= past_target_date) & (~price_df[stock_code].isna())]

        if len(past_dates) == 0:
            percentage_changes[label] = "Không có dữ liệu"
            continue

        past_date = past_dates[-1]
        past_price = price_df.loc[past_date, stock_code]

        # Tính phần trăm thay đổi
        change = ((current_price - past_price) / past_price) * 100
        percentage_changes[label] = round(change, 2)

    return percentage_changes

def draw_profitability_chart(ratio_df, stock_code):
    df_plot = ratio_df[(ratio_df["Mã"] == stock_code) & (ratio_df["Năm"].between(2020, 2024))].sort_values("Năm")
    if df_plot.empty:
        return None

    buffer = BytesIO()
    plt.figure(figsize=(9, 4.5))

    # Vẽ các đường ROA, ROE, ROS với các tone xanh lá
    plt.plot(df_plot["Năm"], df_plot["ROA (%)"], marker='o', label="ROA (%)", color="#2f9e44", linewidth=2)
    plt.plot(df_plot["Năm"], df_plot["ROE (%)"], marker='o', label="ROE (%)", color="#69db7c", linewidth=2)
    plt.plot(df_plot["Năm"], df_plot["ROS (%)"], marker='o', label="ROS (%)", color="#d45c4c", linewidth=2)

    # Giao diện
    plt.xticks(df_plot["Năm"].astype(int), fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel("Năm", fontsize=10)
    plt.ylabel("Tỷ lệ (%)", fontsize=10)
    plt.title(f"Hiệu quả sinh lời của {stock_code}", fontsize=12, fontweight='bold', color="#2f9e44")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig(buffer, format="png", dpi=150)
    plt.close()

    buffer.seek(0)
    return buffer

def draw_valuation_chart(ratio_df, industry_avg_df, stock_code):
    df_company = ratio_df[(ratio_df["Mã"] == stock_code) & (ratio_df["Năm"].between(2020, 2024))]
    industry_name = df_company["Ngành ICB - cấp 3"].iloc[0] if not df_company.empty else None
    df_industry = industry_avg_df[
        (industry_avg_df["Ngành ICB - cấp 3"] == industry_name) &
        (industry_avg_df["Năm"].between(2020, 2024))
    ]

    if df_company.empty or df_industry.empty:
        return None

    years = sorted(df_company["Năm"].unique())
    pe_company = df_company.set_index("Năm")["P/E"]
    pb_company = df_company.set_index("Năm")["P/B"]
    pe_industry = df_industry.set_index("Năm")["P/E"]
    pb_industry = df_industry.set_index("Năm")["P/B"]

    x = range(len(years))
    width = 0.18

    buffer = BytesIO()
    plt.figure(figsize=(10, 5))

    # Bar chart - tone xanh lá
    plt.bar([i - width*1.5 for i in x], [pe_company.get(y, 0) for y in years], width=width,
            label='P/E - Công ty', color="#12a32a")
    plt.bar([i - width/2 for i in x], [pe_industry.get(y, 0) for y in years], width=width,
            label='P/E - TB ngành', color="#69db7c")
    plt.bar([i + width/2 for i in x], [pb_company.get(y, 0) for y in years], width=width,
            label='P/B - Công ty', color="#bd3b2f")
    plt.bar([i + width*1.5 for i in x], [pb_industry.get(y, 0) for y in years], width=width,
            label='P/B - TB ngành', color="#d16a60")

    # Giao diện
    plt.xticks(x, years)
    plt.ylabel("Tỷ số định giá")
    plt.xlabel("Năm")
    plt.title(f"Tỷ số định giá của {stock_code}", fontsize=12, fontweight='bold', color="#2f9e44")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig(buffer, format="png", dpi=150)
    plt.close()
    buffer.seek(0)
    return buffer

def draw_growth_chart(ratio_df, stock_code):
    df_plot = ratio_df[(ratio_df["Mã"] == stock_code) & (ratio_df["Năm"].between(2020, 2024))].sort_values("Năm")

    if df_plot.empty:
        return None

    buffer = BytesIO()
    plt.figure(figsize=(9, 4.5))

    # Vẽ biểu đồ với tone xanh lá
    plt.plot(df_plot["Năm"], df_plot["Revenue Growth (%)"], marker='o',
             label="Tăng trưởng Doanh thu (%)", color="#69db7c", linewidth=2)
    plt.plot(df_plot["Năm"], df_plot["Net Income Growth (%)"], marker='o',
             label="Tăng trưởng LNST (%)", color="#2f9e44", linewidth=2)

    # Định dạng trục và giao diện
    plt.xticks(df_plot["Năm"].astype(int), fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel("Năm", fontsize=10)
    plt.ylabel("Tăng trưởng (%)", fontsize=10)
    plt.title(f"Tăng trưởng doanh thu và lợi nhuận của {stock_code}", fontsize=12, fontweight='bold', color="#2f9e44")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()

    # Xuất buffer hình
    plt.savefig(buffer, format="png", dpi=150)
    plt.close()
    buffer.seek(0)
    return buffer

def draw_leverage_chart(ratio_df, stock_code):
    # Lọc dữ liệu theo mã và năm từ 2020 đến 2024
    df_plot = ratio_df[(ratio_df["Mã"] == stock_code) & (ratio_df["Năm"].between(2020, 2024))].sort_values("Năm")

    if df_plot.empty:
        return None

    buffer = BytesIO()
    plt.figure(figsize=(9, 4.5))

    # Vẽ biểu đồ với tone xanh lá
    plt.plot(df_plot["Năm"], df_plot["D/A (%)"], marker='o', label="D/A (%)", color="#8ce99a", linewidth=2)
    plt.plot(df_plot["Năm"], df_plot["D/E (%)"], marker='o', label="D/E (%)", color="#2f9e44", linewidth=2)
    plt.plot(df_plot["Năm"], df_plot["E/A (%)"], marker='o', label="E/A (%)", color="#d13a2c", linewidth=2)

    # Cài đặt trục và giao diện
    plt.xticks(df_plot["Năm"].astype(int), fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel("Năm", fontsize=10)
    plt.ylabel("Tỷ lệ (%)", fontsize=10)
    plt.title(f"Đòn bẩy tài chính của {stock_code}", fontsize=12, fontweight='bold', color="#2f9e44")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()

    # Lưu biểu đồ vào buffer
    plt.savefig(buffer, format="png", dpi=150)
    plt.close()
    buffer.seek(0)

    return buffer

def draw_asset_liability_chart(bcdkt_df, stock_code):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from io import BytesIO

    # Chuẩn hóa dữ liệu
    df = bcdkt_df.copy()
    df["Năm"] = pd.to_numeric(df["Năm"], errors="coerce")
    df = df[df["Mã"] == stock_code].sort_values("Năm")

    years = df["Năm"].astype(int).tolist()
    assets_short = df["TÀI SẢN NGẮN HẠN"].tolist()
    assets_long = df["TÀI SẢN DÀI HẠN"].tolist()
    liabilities = df["NỢ PHẢI TRẢ"].tolist()
    equity = df["VỐN CHỦ SỞ HỮU"].tolist()

    def fmt_thousands(x, pos):
        return f"{int(x / 1_000_000):,}".replace(",", ".")  # Đơn vị: triệu -> hàng tỷ

    fig, axs = plt.subplots(2, 3, figsize=(10, 5))  # 2 hàng x 3 cột
    bar_width = 0.4

    # Tài sản ngắn hạn
    axs[0, 0].bar(years, assets_short, color="#b2f2bb", width=bar_width)
    axs[0, 0].set_title("Tài sản ngắn hạn", fontsize=10)
    axs[0, 0].yaxis.set_major_formatter(FuncFormatter(fmt_thousands))

    # Tài sản dài hạn
    axs[0, 1].bar(years, assets_long, color="#69db7c", width=bar_width)
    axs[0, 1].set_title("Tài sản dài hạn", fontsize=10)
    axs[0, 1].yaxis.set_major_formatter(FuncFormatter(fmt_thousands))

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # Tổng tài sản - Donut Chart
    latest_total_assets = int(assets_short[-1] + assets_long[-1]) // 1_000_000
    # Tạo trục con (inset axis) nằm trong axs[0, 2] nhưng kiểm soát kích thước tốt hơn
    ax_donut1 = inset_axes(axs[0, 2], width="100%", height="100%", loc='center')

    ax_donut1.pie(
        [assets_short[-1], assets_long[-1]],
        colors=["#b2f2bb", "#69db7c"],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.4),
    )
    ax_donut1.text(
        0, 0,
        f"Tổng tài sản\n{latest_total_assets:,}".replace(",", "."),
        ha="center", va="center", fontsize=10, weight="bold"
    )
    ax_donut1.set_aspect('equal')
    axs[0, 2].axis("off")  # Ẩn trục gốc nếu không cần khung viền

    # Nợ phải trả
    axs[1, 0].bar(years, liabilities, color="#ffa8a8", width=bar_width)
    axs[1, 0].set_title("Nợ phải trả", fontsize=10)
    axs[1, 0].yaxis.set_major_formatter(FuncFormatter(fmt_thousands))

    # Vốn chủ sở hữu
    axs[1, 1].bar(years, equity, color="#ff6b6b", width=bar_width)
    axs[1, 1].set_title("Vốn chủ sở hữu", fontsize=10)
    axs[1, 1].yaxis.set_major_formatter(FuncFormatter(fmt_thousands))

    # Tổng nguồn vốn - Donut Chart
    latest_total_equity = int(liabilities[-1] + equity[-1]) // 1_000_000

    # Tạo trục con để vẽ biểu đồ donut với kích thước lớn hơn
    ax_donut2 = inset_axes(axs[1, 2], width="100%", height="100%", loc='center')

    ax_donut2.pie(
        [liabilities[-1], equity[-1]],
        colors=["#ffa8a8", "#fa5252"],  # đỏ nhạt và đỏ đậm
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.4),
    )
    ax_donut2.text(
        0, 0,
        f"Tổng nguồn vốn\n{latest_total_equity:,}".replace(",", "."),
        ha="center", va="center", fontsize=10, weight="bold"
    )
    ax_donut2.set_aspect('equal')
    axs[1, 2].axis("off")  # Ẩn trục ngoài nếu không cần khung viền

    # Bỏ trục không cần
    for ax in axs.flat:
        ax.tick_params(axis='x', labelrotation=45)
        if ax != axs[0, 2] and ax != axs[1, 2]:
            ax.set_xlabel("")
            ax.set_ylabel("")

    plt.tight_layout()

    # Xuất thành ảnh buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", transparent=False)
    plt.close()
    buf.seek(0)
    return buf

def add_page_footer(c, width):
    c.setFont("Roboto_Regular", 11)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2, 20, f"Trang {c.getPageNumber()}")

def add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH):
    """Vẽ tiêu đề đầu trang PDF: tên công ty, giá cổ phiếu, ngày báo cáo"""
    x_left = 40
    y_top = height - 40

    # Tên công ty
    c.setFont("Roboto_Black", 18)
    c.setFillColor(GREEN_TEXT)
    c.drawString(x_left, y_top, ten_cong_ty)

    # Giá đóng cửa
    c.setFont("Roboto_Black", 12)
    c.setFillColor(colors.black)
    if isinstance(stock_price, (int, float)):
        gia = f"Giá đóng cửa: {int(stock_price):,} VND"
    else:
        gia = f"Giá đóng cửa: {stock_price}"
    c.drawString(x_left, y_top - 20, gia)

    # Ngày báo cáo
    c.setFont("Roboto_Regular", 12)  # Dùng font thường thay vì đậm
    c.setFillColor(colors.black)
    c.drawString(x_left, y_top - 40, f"Ngày báo cáo: {ngay_tao}")

    # Logo ở góc phải trên cùng (nếu có)
    if logo_path and os.path.exists(logo_path):
        try:
            logo = ImageReader(logo_path)
            logo_width = 60
            logo_height = 60

        # Vị trí sát góc phải trên
            x_logo = width - 40 - logo_width
            y_logo = height - 20 - logo_height  # Đẩy cao hơn một chút
            c.drawImage(logo, x_logo, y_logo, width=logo_width, height=logo_height,
                        preserveAspectRatio=True, mask='auto')
        except:
            print("⚠️ Không thể hiển thị logo.")

    # Gạch ngang dưới header
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 90, width - 40, height - 90)

def generate_price_chart_comment(price_df, stock_code, selected_date):
    selected_date = pd.to_datetime(selected_date)
    # Lấy chuỗi giá theo mã
    if stock_code not in price_df.columns:
        return "Không có dữ liệu.", "Không có dữ liệu."

    price_series = price_df[stock_code].dropna()
    price_series = price_series[price_series.index <= pd.to_datetime(selected_date)]

    if len(price_series) < 10:
        return "Không đủ dữ liệu.", "Không đủ dữ liệu."

    # --- Nhận xét 6 tháng ---
    start_6m = selected_date - timedelta(days=183)
    prices_6m = price_series[price_series.index >= start_6m]
    if not prices_6m.empty:
        change_6m = (prices_6m.iloc[-1] - prices_6m.iloc[0]) / prices_6m.iloc[0] * 100
        change_6m = round(change_6m, 2)
        comment_6m = (
            f"Từ thời điểm cách đây 6 tháng đến ngày {selected_date.strftime('%d/%m/%Y')}, "
            f"giá cổ phiếu thay đổi khoảng {change_6m:+.2f}%. "
        )
        if change_6m > 10:
            comment_6m += "Xu hướng tăng trưởng tích cực trong ngắn hạn phản ánh sự kỳ vọng của thị trường đối với doanh nghiệp."
        elif change_6m < -10:
            comment_6m += "Giá cổ phiếu có dấu hiệu suy giảm rõ rệt trong ngắn hạn, có thể do biến động thị trường hoặc kết quả kinh doanh không khả quan."
        else:
            comment_6m += "Giá cổ phiếu dao động nhẹ, phản ánh tâm lý thị trường đang chờ đợi thêm tín hiệu rõ ràng từ doanh nghiệp."
    else:
        comment_6m = "Không đủ dữ liệu 6 tháng gần nhất để đưa ra nhận xét."

    # --- Nhận xét 5 năm ---
    start_5y = selected_date - timedelta(days=5 * 365)
    prices_5y = price_series[price_series.index >= start_5y]
    if not prices_5y.empty:
        change_5y = (prices_5y.iloc[-1] - prices_5y.iloc[0]) / prices_5y.iloc[0] * 100
        change_5y = round(change_5y, 2)
        comment_5y = (
            f"Trong giai đoạn 5 năm qua, giá cổ phiếu thay đổi khoảng {change_5y:+.2f}%. "
        )
        if change_5y > 30:
            comment_5y += "Điều này cho thấy xu hướng tăng trưởng dài hạn ổn định và tích cực, phù hợp với kỳ vọng mở rộng quy mô và tăng trưởng lợi nhuận của công ty."
        elif change_5y < -30:
            comment_5y += "Xu hướng giảm trong dài hạn có thể là tín hiệu tiêu cực, phản ánh những thách thức lớn về hoạt động hoặc cạnh tranh trong ngành."
        else:
            comment_5y += "Giá cổ phiếu biến động nhẹ trong dài hạn, cho thấy mức độ ổn định nhất định hoặc thiếu động lực tăng trưởng rõ ràng."
    else:
        comment_5y = "Không đủ dữ liệu trong 5 năm để đưa ra nhận xét."

    return comment_6m, comment_5y

def generate_financial_commentary(bcdkt_df, stock_code):
    df = bcdkt_df[bcdkt_df['Mã'] == stock_code].copy()
    if df.empty:
        return "Báo cáo tài chính cho thấy doanh nghiệp đang duy trì hoạt động ổn định trong giai đoạn gần đây."

    # Không cần xử lý số liệu chi tiết, chỉ kiểm tra số năm
    years = sorted(df["Năm"].dropna().astype(int).unique())
    num_years = len(years)

    if num_years >= 3:
        return (
            "Doanh nghiệp duy trì tăng trưởng ổn định về tài sản và vốn chủ sở hữu trong những năm gần đây. "
            "Tỷ lệ nợ có thể dao động nhưng vẫn nằm trong mức kiểm soát. "
            "Cơ cấu tài chính được duy trì hợp lý, phản ánh năng lực hoạt động bền vững."
        )
    elif num_years == 2:
        return (
            "Báo cáo tài chính cho thấy doanh nghiệp có sự ổn định trong cơ cấu nguồn vốn. "
            "Tài sản và vốn chủ sở hữu duy trì ở mức tương đối, giúp đảm bảo khả năng thanh toán ngắn hạn."
        )
    else:
        return (
            "Báo cáo tài chính thể hiện quy mô doanh nghiệp ở mức vừa phải, với cơ cấu tài sản và nguồn vốn đơn giản. "
            "Cần theo dõi thêm dữ liệu các năm tiếp theo để đánh giá xu hướng dài hạn."
        )
 
def generate_income_commentary(kqkd_df, stock_code):
    df = kqkd_df[kqkd_df["Mã"] == stock_code].copy()
    if df.empty:
        return "Doanh nghiệp duy trì hoạt động kinh doanh ổn định, với kết quả tài chính phù hợp theo từng giai đoạn."

    df = df.sort_values("Năm")
    years = df["Năm"].dropna().astype(int).tolist()

    try:
        y_start, y_end = years[0], years[-1]
        dt_start = df[df["Năm"] == y_start]["Doanh thu thuần"].values[0]
        dt_end = df[df["Năm"] == y_end]["Doanh thu thuần"].values[0]

        ln_start = df[df["Năm"] == y_start]["Lợi nhuận sau thuế thu nhập doanh nghiệp"].values[0]
        ln_end = df[df["Năm"] == y_end]["Lợi nhuận sau thuế thu nhập doanh nghiệp"].values[0]

        def gen_comment(label, start, end):
            if start == 0:
                return f"{label} có xu hướng tăng nhẹ"
            change = (end - start) / abs(start) * 100
            if change > 20:
                return f"{label} tăng mạnh (+{change:.1f}%)"
            elif change > 5:
                return f"{label} tăng nhẹ (+{change:.1f}%)"
            elif change < -20:
                return f"{label} giảm mạnh ({change:.1f}%)"
            elif change < -5:
                return f"{label} giảm nhẹ ({change:.1f}%)"
            else:
                return f"{label} ổn định"

        dt_text = gen_comment("doanh thu thuần", dt_start, dt_end)
        ln_text = gen_comment("Lợi nhuận sau thuế", ln_start, ln_end)

        summary = (
            f"Từ năm {y_start} đến {y_end}, {dt_text}, phản ánh hiệu quả bán hàng và hoạt động chính. "
            f"{ln_text}, cho thấy hiệu quả kinh doanh tổng thể của doanh nghiệp có xu hướng {'cải thiện' if ln_end > ln_start else 'suy giảm' if ln_end < ln_start else 'duy trì ổn định'}."
        )

        return summary

    except Exception:
        return "Hoạt động kinh doanh của doanh nghiệp có sự thay đổi theo từng năm, phản ánh tính chu kỳ và ảnh hưởng của thị trường. Nên theo dõi thêm để có đánh giá chính xác hơn."
 
def generate_cashflow_commentary(lctt_df, stock_code):
    df = lctt_df[lctt_df["Mã"] == stock_code].copy()
    if df.empty:
        return "Lưu chuyển tiền tệ của doanh nghiệp phản ánh tình hình tài chính đang được kiểm soát tốt qua các năm."

    df = df.sort_values("Năm")
    years = df["Năm"].dropna().astype(int).tolist()

    try:
        y_start, y_end = years[0], years[-1]

        lc_start = df[df["Năm"] == y_start]["Lưu chuyển tiền thuần trong kỳ (TT)"].values[0]
        lc_end = df[df["Năm"] == y_end]["Lưu chuyển tiền thuần trong kỳ (TT)"].values[0]

        cash_start = df[df["Năm"] == y_start]["Tiền và tương đương tiền cuối kỳ (TT)"].values[0]
        cash_end = df[df["Năm"] == y_end]["Tiền và tương đương tiền cuối kỳ (TT)"].values[0]

        def gen_change_text(label, start, end):
            if start == 0:
                return f"{label} có sự biến động"
            change = (end - start) / abs(start) * 100
            if change > 20:
                return f"{label} tăng mạnh (+{change:.1f}%)"
            elif change > 5:
                return f"{label} tăng nhẹ (+{change:.1f}%)"
            elif change < -20:
                return f"{label} giảm mạnh ({change:.1f}%)"
            elif change < -5:
                return f"{label} giảm nhẹ ({change:.1f}%)"
            else:
                return f"{label} ổn định"

        flow_text = gen_change_text("dòng tiền thuần", lc_start, lc_end)
        cash_text = gen_change_text("Tiền cuối kỳ", cash_start, cash_end)

        return (
            f"Từ năm {y_start} đến {y_end}, {flow_text}, phản ánh khả năng tạo ra dòng tiền từ hoạt động của doanh nghiệp. "
            f"{cash_text}, cho thấy mức độ an toàn tài chính và thanh khoản của công ty được duy trì hợp lý."
        )

    except Exception:
        return "Lưu chuyển tiền tệ biến động theo từng năm. Doanh nghiệp cần tiếp tục kiểm soát dòng tiền để duy trì thanh khoản ổn định."

def generate_asset_liability_commentary(bcdkt_df, stock_code: str) -> str:
    df = bcdkt_df[bcdkt_df["Mã"] == stock_code].copy()
    df = df.sort_values("Năm")

    if df.empty:
        return "Doanh nghiệp duy trì cơ cấu tài sản và nguồn vốn ổn định trong các năm gần đây."

    try:
        latest = df.iloc[-1]
        short_term = latest.get("TÀI SẢN NGẮN HẠN", 0)
        long_term = latest.get("TÀI SẢN DÀI HẠN", 0)
        liabilities = latest.get("NỢ PHẢI TRẢ", 0)
        equity = latest.get("VỐN CHỦ SỞ HỮU", 0)

        total_assets = short_term + long_term
        total_funding = liabilities + equity

        pct_short = short_term / total_assets * 100 if total_assets else 0
        pct_long = long_term / total_assets * 100 if total_assets else 0
        pct_debt = liabilities / total_funding * 100 if total_funding else 0
        pct_equity = equity / total_funding * 100 if total_funding else 0

        # Nhận xét tài sản
        if pct_short > pct_long:
            asset_comment = f"Tài sản ngắn hạn chiếm tỷ trọng lớn ({pct_short:.1f}%), cho thấy doanh nghiệp có tính thanh khoản tốt."
        else:
            asset_comment = f"Tài sản dài hạn chiếm tỷ trọng lớn ({pct_long:.1f}%), phản ánh định hướng đầu tư dài hạn của doanh nghiệp."

        # Nhận xét nguồn vốn
        if pct_equity >= 50:
            funding_comment = f"Vốn chủ sở hữu chiếm ưu thế ({pct_equity:.1f}%), cho thấy cấu trúc tài chính an toàn."
        else:
            funding_comment = f"Nợ phải trả chiếm tỷ trọng cao ({pct_debt:.1f}%), thể hiện doanh nghiệp đang sử dụng đòn bẩy tài chính."

        return f"{asset_comment} {funding_comment}"

    except Exception:
        return "Doanh nghiệp duy trì cơ cấu tài sản và nguồn vốn ổn định trong các năm gần đây."

def generate_summary_data(ratio_df, industry_df, lctt_df, stock_code):
    import numpy as np

    summary = {}

    df_company = ratio_df[(ratio_df["Mã"] == stock_code) & (ratio_df["Năm"].between(2020, 2024))]
    if df_company.empty:
        return {}

    # Lấy tên ngành
    industry = df_company["Ngành ICB - cấp 3"].iloc[0]
    df_industry = industry_df[(industry_df["Ngành ICB - cấp 3"] == industry) & (industry_df["Năm"].between(2020, 2024))]

    # ===== Doanh thu và lợi nhuận =====
    rev_growth = df_company["Revenue Growth (%)"].dropna()
    net_growth = df_company["Net Income Growth (%)"].dropna()

    if not rev_growth.empty:
        avg_rev = rev_growth.mean()
        if avg_rev > 15:
            summary["revenue_trend"] = "tăng mạnh"
        elif avg_rev > 5:
            summary["revenue_trend"] = "tăng nhẹ"
        elif avg_rev < -5:
            summary["revenue_trend"] = "giảm"
        else:
            summary["revenue_trend"] = "ổn định"

    if not net_growth.empty:
        std_net = net_growth.std()
        if std_net > 20:
            summary["profit_trend"] = "dao động mạnh"
        elif std_net > 10:
            summary["profit_trend"] = "dao động nhẹ"
        else:
            summary["profit_trend"] = "ổn định"

    # ===== ROE / ROA =====
    if not df_company.empty and not df_industry.empty:
        roe_cmp = df_company.groupby("Năm")["ROE (%)"].mean()
        roe_ind = df_industry.groupby("Năm")["ROE (%)"].mean()
        roa_cmp = df_company.groupby("Năm")["ROA (%)"].mean()
        roa_ind = df_industry.groupby("Năm")["ROA (%)"].mean()

        if (roe_cmp > roe_ind).sum() >= 3:
            summary["roe_status"] = "cao hơn trung bình ngành"
        elif (roe_cmp < roe_ind).sum() >= 3:
            summary["roe_status"] = "thấp hơn trung bình ngành"
        else:
            summary["roe_status"] = "gần bằng trung bình ngành"

        if (roa_cmp > roa_ind).sum() >= 3:
            summary["roa_status"] = "cao hơn trung bình ngành"
        elif (roa_cmp < roa_ind).sum() >= 3:
            summary["roa_status"] = "thấp hơn trung bình ngành"
        else:
            summary["roa_status"] = "trung bình"

    # ===== Đòn bẩy tài chính =====
    de_ratio = df_company["D/E (%)"].dropna()
    if not de_ratio.empty:
        avg_de = de_ratio.mean()
        if avg_de > 120:
            summary["debt_ratio"] = "cao"
        elif avg_de < 60:
            summary["debt_ratio"] = "thấp"
        else:
            summary["debt_ratio"] = "vừa phải"

    # ===== Cảnh báo dòng tiền =====
    lctt_stock = lctt_df[lctt_df["Mã"] == stock_code]
    if not lctt_stock.empty:
        net_cash_flows = lctt_stock.groupby("Năm")["Lưu chuyển tiền thuần trong kỳ (TT)"].sum()
        negative_years = (net_cash_flows < 0).sum()
        summary["cashflow_warning"] = negative_years >= 2

    # ===== Định giá =====
    pe_cmp = df_company.groupby("Năm")["P/E"].mean()
    pe_ind = df_industry.groupby("Năm")["P/E"].mean()
    pb_cmp = df_company.groupby("Năm")["P/B"].mean()
    pb_ind = df_industry.groupby("Năm")["P/B"].mean()

    pe_better = (pe_cmp < pe_ind).sum() >= 3
    pb_better = (pb_cmp < pb_ind).sum() >= 3

    if pe_better and pb_better:
        summary["valuation_comment"] = "định giá thấp hơn ngành"
    elif not pe_better and not pb_better:
        summary["valuation_comment"] = "định giá cao hơn ngành"
    else:
        summary["valuation_comment"] = "định giá tương đương ngành"

    # ===== Triển vọng đầu tư =====
    if summary.get("revenue_trend") in ["tăng mạnh", "tăng nhẹ"] and summary.get("roe_status") == "cao hơn trung bình ngành":
        summary["investment_potential"] = "Tăng trưởng ổn định trong dài hạn"
    else:
        summary["investment_potential"] = "Cần theo dõi thêm các yếu tố cơ bản"

    # ===== Rủi ro tổng hợp =====
    if summary.get("cashflow_warning") or summary.get("debt_ratio") == "cao":
        summary["risk_warning"] = "Có dấu hiệu rủi ro tài chính cần theo dõi"
    else:
        summary["risk_warning"] = "Không có dấu hiệu rủi ro tài chính nghiêm trọng"

    return summary

def generate_investment_recommendation(summary: dict) -> str:
    """
    Sinh đoạn khuyến nghị đầu tư dài dựa trên kết quả phân tích tổng hợp (summary).
    """
    lines = []

    # Phân tích tài chính
    lines.append("\nPHÂN TÍCH TÀI CHÍNH:")

    # Doanh thu
    trend = summary.get("revenue_trend", "")
    if "tăng mạnh" in trend:
        lines.append("- Doanh thu tăng trưởng mạnh mẽ trong giai đoạn gần đây, phản ánh xu hướng mở rộng hoạt động tích cực.")
    elif "tăng nhẹ" in trend:
        lines.append("- Doanh thu có xu hướng tăng trưởng nhẹ, cho thấy công ty vẫn đang giữ được đà phát triển ổn định.")
    elif "giảm" in trend:
        lines.append("- Doanh thu có xu hướng giảm, điều này có thể phản ánh sự sụt giảm về nhu cầu thị trường hoặc hiệu quả kinh doanh.")
    else:
        lines.append("- Doanh thu duy trì ở mức ổn định qua các năm.")

    # Lợi nhuận
    profit_trend = summary.get("profit_trend", "")
    if "dao động mạnh" in profit_trend:
        lines.append("- Lợi nhuận sau thuế dao động mạnh, cho thấy tính ổn định chưa cao trong hiệu quả hoạt động.")
    elif "dao động nhẹ" in profit_trend:
        lines.append("- Lợi nhuận có sự dao động nhẹ, tuy nhiên vẫn giữ được xu hướng tích cực.")
    else:
        lines.append("- Lợi nhuận duy trì ổn định, thể hiện sự kiểm soát tốt trong chi phí và vận hành.")

    # ROE, ROA
    roe_cmp = summary.get("roe_vs_industry", "")
    roa_cmp = summary.get("roa_vs_industry", "")
    if "thấp" in roe_cmp and "thấp" in roa_cmp:
        lines.append("- Chỉ số ROE và ROA thấp hơn trung bình ngành, phản ánh hiệu quả sử dụng vốn và tài sản chưa thực sự nổi bật.")
    elif "cao" in roe_cmp or "cao" in roa_cmp:
        lines.append("- ROE hoặc ROA cao hơn trung bình ngành, cho thấy khả năng tạo lợi nhuận tốt trên vốn và tài sản.")
    else:
        lines.append("- ROE và ROA tương đương ngành, phản ánh hiệu quả hoạt động ở mức trung bình.")

    # Tỷ lệ nợ
    leverage = summary.get("de_ratio_level", "")
    if leverage == "cao":
        lines.append("- Tỷ lệ nợ trên vốn chủ sở hữu cao, điều này cần được theo dõi do có thể làm gia tăng rủi ro tài chính.")
    elif leverage == "thấp":
        lines.append("- Tỷ lệ nợ thấp, thể hiện cấu trúc tài chính an toàn.")
    else:
        lines.append("- Tỷ lệ nợ ở mức hợp lý so với ngành.")

    # Cảnh báo dòng tiền
    if summary.get("cashflow_warning", False):
        lines.append("- Dòng tiền hoạt động kinh doanh âm trong nhiều năm, điều này cần được lưu ý vì có thể ảnh hưởng đến khả năng thanh toán ngắn hạn.")
    else:
        lines.append("- Dòng tiền hoạt động ổn định, hỗ trợ tốt cho hoạt động kinh doanh.")

    # Định giá
    pe = summary.get("pe_valuation", "")
    pb = summary.get("pb_valuation", "")
    if "cao hơn" in pe or "cao hơn" in pb:
        lines.append("- Định giá cổ phiếu hiện cao hơn trung bình ngành, nhà đầu tư cần cân nhắc về mức định giá trước khi ra quyết định.")
    elif "thấp hơn" in pe or "thấp hơn" in pb:
        lines.append("- Cổ phiếu đang được định giá thấp hơn trung bình ngành, có thể là cơ hội nếu các yếu tố cơ bản được cải thiện.")
    else:
        lines.append("- Định giá cổ phiếu tương đương với trung bình ngành.")

    # Đánh giá triển vọng
    lines.append("\nĐÁNH GIÁ TRIỂN VỌNG:")
    lines.append(summary.get("investment_outlook", "Doanh nghiệp có tiềm năng tăng trưởng nếu duy trì được hiệu quả và kiểm soát tốt rủi ro."))

    # Rủi ro tổng hợp
    if summary.get("overall_risk", "thấp") == "cao":
        lines.append("Tuy nhiên, nhà đầu tư cần thận trọng do mức rủi ro tổng thể đang ở mức cao.")

    # Kết luận
    lines.append("\nKẾT LUẬN:")
    lines.append(summary.get("final_comment", "Cổ phiếu phù hợp với nhà đầu tư trung lập hoặc tích cực, tùy vào khẩu vị rủi ro."))

    return "\n".join(lines)

def create_pdf(stock_code, report_date):
    """Tạo file PDF chứa thông tin doanh nghiệp theo mã đã chọn."""
    stock_info = df[df["Mã"] == stock_code]
    if stock_info.empty:
        return None

    # Lấy thông tin
    ten_cong_ty = stock_info.iloc[0]["Tên công ty"]
    san = stock_info.iloc[0]["Sàn"]
    nganh_cap1 = stock_info.iloc[0]["Ngành ICB - cấp 1"]
    nganh_cap2 = stock_info.iloc[0]["Ngành ICB - cấp 2"]
    nganh_cap3 = stock_info.iloc[0]["Ngành ICB - cấp 3"]
    nganh_cap4 = stock_info.iloc[0]["Ngành ICB - cấp 4"]
    ngay_tao = report_date.strftime('%d-%m-%Y')

    # Lấy thông tin tóm tắt doanh nghiệp
    info_data = info_df[info_df["Mã CK"] == stock_code]
    tom_tat = info_data["Thông tin"].values[0] if not info_data.empty else "Không có thông tin."

    # Lấy giá cổ phiếu
    report_date = pd.to_datetime(selected_date)  # Đảm bảo kiểu datetime64
    if report_date in price_df.index and stock_code in price_df.columns:
        stock_price = price_df.loc[report_date, stock_code]
    else:
        stock_price = "Không có dữ liệu"

    # Tính giá cao nhất và thấp nhất trong 52 tuần trước ngày báo cáo
    start_date_52_weeks = report_date - timedelta(weeks=52)
    stock_price_52_weeks = price_df[stock_code].loc[start_date_52_weeks:report_date]

    highest_52_weeks = stock_price_52_weeks.max() if not stock_price_52_weeks.empty else "Không có dữ liệu"
    lowest_52_weeks = stock_price_52_weeks.min() if not stock_price_52_weeks.empty else "Không có dữ liệu"

    # Lấy thông tin SLCP lưu hành từ ratio.xlsx
    ratio_df['Năm'] = pd.to_numeric(ratio_df['Năm'], errors='coerce')
    ratio_df['Mã'] = ratio_df['Mã'].astype(str).str.strip()
    ratio_data = ratio_df[(ratio_df['Mã'] == stock_code) & (ratio_df['Năm'] == selected_date.year)]
    slcp = ratio_data['SLCP lưu hành'].values[0] if not ratio_data.empty else "Không có dữ liệu"

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    add_page_footer(c, width)
    c.setFont("Roboto_Black", 16)

    # Tiêu đề
    add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH)

    # Tiêu đề "Thông tin chung"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 107, "THÔNG TIN CHUNG VỀ DOANH NGHIỆP")

    # Dữ liệu bảng
    data = [
        ["Mã chứng khoán", stock_code],
        ["Tên công ty", ten_cong_ty],
        ["Sàn chứng khoán", san],
        ["Ngành", f"{nganh_cap1} - {nganh_cap2} - {nganh_cap3} - {nganh_cap4}"]
    ]

    # Tạo bảng
    table = Table(data, colWidths=[100, width - 180])
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Roboto_Black'),
        ('FONTNAME', (1, 0), (1, -1), 'Roboto_Regular'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),

        # Màu nền xen kẽ theo hàng: hàng 2 & 4 xanh, hàng 1 & 3 trắng
        ('BACKGROUND', (0, 1), (-1, 1), LIGHT_GREEN_BG),  # Hàng thứ 2
        ('BACKGROUND', (0, 3), (-1, 3), LIGHT_GREEN_BG),  # Hàng thứ 4
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),  # Hàng thứ 1
        ('BACKGROUND', (0, 2), (-1, 2), colors.white),  # Hàng thứ 3
    ]))

    # Vẽ bảng lên PDF
    table.wrapOn(c, width, height)
    table.drawOn(c, 40, height - 205)

    # Đường kẻ xanh phía dưới bảng
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 125 - len(data) * 22, width - 40, height - 125 - len(data) * 22)

    # Tiêu đề "TỔNG QUAN VỀ DOANH NGHIỆP"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 145 - len(data) * 22, "TỔNG QUAN VỀ DOANH NGHIỆP")

    # Hiển thị nội dung tóm tắt doanh nghiệp
    styles = getSampleStyleSheet()
    styleN = ParagraphStyle(
        'Normal',
        parent=styles["Normal"],
        fontName="Roboto_Regular",  # Sử dụng font đã đăng ký
        fontSize=11,
        leading=15,  # Điều chỉnh khoảng cách
        alignment=TA_JUSTIFY
    )

    p = Paragraph(tom_tat, styleN)

    # Xác định vị trí và kích thước vùng văn bản
    w, h = p.wrap(width - 80, height - 240)
    p.drawOn(c, 40, height - 240 - h)

    # Vẽ đường kẻ xanh dương dưới đoạn văn bản
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 250 - h, width - 40, height - 250 - h)

    # Tiêu đề "BIỂU ĐỒ GIÁ TRỊ VỐN HOÁ THỊ TRƯỜNG"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 260 - len(data) * 22, "BIỂU ĐỒ GIÁ TRỊ VỐN HOÁ THỊ TRƯỜNG")
    c.setFont("Roboto_Regular", 10)
    c.drawString(40, height - 365, "(Đơn vị: tỷ VND)")

    # Biểu đồ market cap
    market_cap_chart = draw_marketcap_chart(marketcap_df, stock_code)
    if market_cap_chart:
        c.drawImage(ImageReader(market_cap_chart), x=40, y=100, width=500, height=360)
    
    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 750, width - 40, height - 750)

    # Ngắt sang trang mới
    add_page_footer(c, width)
    c.showPage()

    # Tiêu đề
    add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH)

    # Tiêu đề "BIỂU ĐỒ GIÁ"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 37 - h, "BIỂU ĐỒ GIÁ")

    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 120, width - 40, height - 120)

    #Chữ "5 năm"
    c.setFont("Roboto_Regular", 11)
    c.drawString(120, height - 275 - h, "TRONG 5 NĂM")

    #Vẽ biểu đồ
    chart_buffer = plot_stock_price(stock_code)
    if chart_buffer:
        img = Image(chart_buffer, width=240, height=140)
        img.wrapOn(c, width, height)
        img.drawOn(c, 40, height - 420 - h)

    # Chữ "6 tháng"
    c.setFont("Roboto_Regular", 11)
    c.drawString(120, height - 90 - h, "TRONG 6 THÁNG")

    # Vẽ biểu đồ
    chart_buffer = plot_stock_price1(stock_code, selected_date)
    if chart_buffer:
        img = Image(chart_buffer, width=240, height=140)
        img.wrapOn(c, width, height)
        img.drawOn(c, 40, height - 235 - h)

    # Vẽ đường kẻ xanh dương dưới đoạn văn bản
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 435 - h, width - 310, height - 435 - h)

    comment_6m, comment_5y = generate_price_chart_comment(price_df, stock_code, selected_date)
    # Style nhận xét
    style_comment = ParagraphStyle(
        'Comment',
        fontName="Roboto_Regular",
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=colors.black  # Nội dung nhận xét vẫn màu đen
    )
    # Nhận xét biểu đồ 6 tháng
    p1 = Paragraph(
        f'<font color="#1B5E20"><b>Nhận xét:</b></font><br/>{comment_6m}',
        style_comment
    )
    p1.wrapOn(c, 220, height)
    p1.drawOn(c, 320, height - 260)
    # Nhận xét biểu đồ 5 năm
    p2 = Paragraph(
        f'<font color="#1B5E20"><b>Nhận xét:</b></font><br/>{comment_5y}',
        style_comment
    )
    p2.wrapOn(c, 220, height)
    p2.drawOn(c, 320, height - 430)

    # Tiêu đề "THÔNG TIN CỔ PHIẾU"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 455 - h, "THÔNG TIN CỔ PHIẾU")

    # Thêm bảng thông tin nhỏ phía dưới "Thông tin chung"
    small_table_data = [
        ["Giá đóng cửa", f"{int(stock_price):,}"],
        ["52 tuần cao nhất", f"{int(highest_52_weeks):,}" if highest_52_weeks != "Không có dữ liệu" else highest_52_weeks],
        ["52 tuần thấp nhất", f"{int(lowest_52_weeks):,}" if lowest_52_weeks != "Không có dữ liệu" else lowest_52_weeks],
        ["SLCP lưu hành", f"{int(slcp):,}" if slcp != "Không có dữ liệu" else slcp],
        ["Đơn vị tiền tệ", "VND"]
    ]

    # Tạo bảng nhỏ
    small_table = Table(small_table_data, colWidths=[140, width - 490])
    small_table.setStyle(TableStyle([
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Roboto_Black'),
        ('FONTNAME', (1, 0), (1, -1), 'Roboto_Regular'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, 1), LIGHT_GREEN_BG),
        ('BACKGROUND', (0, 2), (-1, 2), colors.white),
        ('BACKGROUND', (0, 3), (-1, 3), LIGHT_GREEN_BG),
        ('BACKGROUND', (0, 4), (-1, 4), colors.white),
    ]))

    # Vẽ bảng nhỏ lên PDF
    small_table.wrapOn(c, width, height)
    small_table.drawOn(c, 40, height - 580 - h)

    # Vẽ đường kẻ xanh dương dưới đoạn văn bản
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(320, height - 435 - h, width - 40, height - 435 - h)

    # Tiêu đề "PHẦN TRĂM THAY ĐỔI GIÁ"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(400, height - 455 - h, "PHẦN TRĂM THAY ĐỔI GIÁ")

    percentage_changes = calculate_percentage_change(price_df, selected_date, stock_code)

    data1 = [
        ["1 ngày", f"{percentage_changes['1 ngày']}%" if isinstance(percentage_changes["1 ngày"], (int, float)) else percentage_changes["1 ngày"]],
        ["5 ngày", f"{percentage_changes['5 ngày']}%" if isinstance(percentage_changes["5 ngày"], (int, float)) else percentage_changes["5 ngày"]],
        ["3 tháng", f"{percentage_changes['3 tháng']}%" if isinstance(percentage_changes["3 tháng"], (int, float)) else percentage_changes["3 tháng"]],
        ["6 tháng", f"{percentage_changes['6 tháng']}%" if isinstance(percentage_changes["6 tháng"], (int, float)) else percentage_changes["6 tháng"]],
        ["1 năm", f"{percentage_changes['1 năm']}%" if isinstance(percentage_changes["1 năm"], (int, float)) else percentage_changes["1 năm"]],
    ]

    table = Table(data1, colWidths=[140, width - 500])
    table.setStyle(TableStyle([
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Căn lề trái cho cột 0
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),  # Căn lề phải cho cột 1
        ('FONTNAME', (0, 0), (0, -1), 'Roboto_Black'),  # Sử dụng phông chữ Roboto_Black cho cột 0
        ('FONTNAME', (1, 0), (1, -1), 'Roboto_Regular'),  # Sử dụng phông chữ Roboto_Regular cho cột 1
        ('FONTSIZE', (0, 0), (-1, -1), 11),  # Cỡ chữ cho toàn bộ bảng
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),  # Khoảng cách giữa các ô trong bảng
        ('TOPPADDING', (0, 0), (-1, -1), 6),  # Khoảng cách trên các ô
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),  # Màu nền cho hàng tiêu đề
        ('BACKGROUND', (0, 1), (-1, 1), LIGHT_GREEN_BG),  # Màu nền cho hàng thứ 1
        ('BACKGROUND', (0, 2), (-1, 2), colors.white),  # Màu nền cho hàng thứ 2
        ('BACKGROUND', (0, 3), (-1, 3), LIGHT_GREEN_BG),  # Màu nền cho hàng thứ 3
        ('BACKGROUND', (0, 4), (-1, 4), colors.white),  # Màu nền cho hàng thứ 4
    ]))

    # Vẽ bảng lên PDF
    table.wrapOn(c, width, height)
    table.drawOn(c, 320, height - 580 - h)

    # Vẽ đường kẻ xanh dương dưới đoạn văn bản
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 595 - h, width - 310, height - 595 - h)

    # Vẽ đường kẻ xanh dương dưới đoạn văn bản
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(320, height - 595 - h, width - 40, height - 595 - h)

    # Tiêu đề "CÁC CHỈ SỐ CƠ BẢN"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 615 - h, "CÁC CHỈ SỐ CƠ BẢN")

    # Đọc dữ liệu ratio theo mã và năm
    ratio_row = ratio_df[(ratio_df['Mã'] == stock_code) & (ratio_df['Năm'] == selected_date.year)]
    kqkd_df = pd.read_csv(os.path.join(DATA_PATH, "KQKD.csv"))

    # Hàm định dạng số
    def fmt(x, is_int=False):
        if isinstance(x, (int, float)):
            return f"{int(round(x)):,}" if is_int else f"{round(x, 2):,}"
        return x

    # Khởi tạo mặc định
    eps_value = "Không có dữ liệu"
    pe = "Không có dữ liệu"
    pb = "Không có dữ liệu"
    book_value = "Không có dữ liệu"

    # Xử lý nếu có dữ liệu
    if not ratio_row.empty:
        row = ratio_row.iloc[0]

        # EPS từ KQKD
        eps_row = kqkd_df[(kqkd_df['Mã'] == stock_code) & (kqkd_df['Năm'] == selected_date.year)]
        if not eps_row.empty:
            eps_value = eps_row["Lãi cơ bản trên cổ phiếu"].values[0]

        # Lấy các chỉ số khác
        pe = row.get("P/E", "Không có dữ liệu")
        pb = row.get("P/B", "Không có dữ liệu")

        # Giá trị sổ sách cần kiểm tra kỹ kiểu dữ liệu
        try:
            book_value = float(row.get("Giá trị sổ sách", None))
        except:
            book_value = "Không có dữ liệu"

    # Tạo bảng hiển thị
    financial_table_data_1 = [["EPS (VND)", fmt(eps_value, is_int=True)]]
    if pe != "Không có dữ liệu":
        financial_table_data_1.append(["P/E", fmt(pe)])

    financial_table_data_2 = [
        ["Giá trị sổ sách (VND)", fmt(book_value, is_int=True)],
        ["P/B", fmt(pb)],
    ]

    table = Table(financial_table_data_1, colWidths=[140, width - 490])
    table.setStyle(TableStyle([
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Roboto_Black'),
        ('FONTNAME', (1, 0), (1, -1), 'Roboto_Regular'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, 1), LIGHT_GREEN_BG),
    ]))

    # Vẽ bảng lên PDF
    table.wrapOn(c, width, height)
    table.drawOn(c, 40, height - 665 - h)

    # Tạo và vẽ bảng
    table2 = Table(financial_table_data_2, colWidths=[140, width - 490])
    table2.setStyle(TableStyle([
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Roboto_Black'),
        ('FONTNAME', (1, 0), (1, -1), 'Roboto_Regular'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, 1), LIGHT_GREEN_BG),
    ]))

    # Vẽ bảng lên PDF
    table2.wrapOn(c, width, height)
    table2.drawOn(c, 320, height - 665 - h)

    # Ngắt sang trang mới
    add_page_footer(c, width)
    c.showPage()

    # Tiêu đề
    add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH)

    # Tiêu đề "Báo cáo tài chính"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 107, "BÁO CÁO TÀI CHÍNH")

    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 112, width - 40, height - 112)

    # Tiêu đề "Bảng cân đối kế toán"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 130, "Bảng cân đối kế toán")

    # Lọc theo mã
    bcdkt_stock = bcdkt_df[bcdkt_df['Mã'] == stock_code]

    # Lấy các năm theo thứ tự tăng dần
    years = sorted(bcdkt_stock['Năm'].dropna().astype(int).unique())

    # Tạo tiêu đề bảng
    headers = ["Chỉ tiêu"] + [str(y) for y in years]

    # Các chỉ tiêu cần hiển thị
    fields = [
        "TÀI SẢN NGẮN HẠN",
        "TÀI SẢN DÀI HẠN",
        "TỔNG CỘNG TÀI SẢN",
        "NỢ PHẢI TRẢ",
        "VỐN CHỦ SỞ HỮU",
        "TỔNG CỘNG NGUỒN VỐN",
    ]

    # Tạo dữ liệu bảng
    data = [headers]
    for field in fields:
        row = [field.replace("_", " ").title()]
        for y in years:
            val = bcdkt_stock.loc[(bcdkt_stock['Năm'] == y), field]
            if not val.empty and pd.notna(val.values[0]):
                value_million = int(val.values[0]) // 1_000_000
                row.append(f"{value_million:,}")
            else:
                row.append("Không có")
        data.append(row)

    # Tạo bảng PDF
    usable_width = width - 80
    colWidths = [250] + [(usable_width - 250) / len(years)] * len(years)
    table = Table(data, colWidths=colWidths)

    # Tạo danh sách dòng có nền xen kẽ (bỏ dòng đầu vì là header)
    background_styles = [('BACKGROUND', (0, 0), (-1, 0), colors.white)]  # Header

    for i in range(1, len(data)):
        bg_color = LIGHT_GREEN_BG if i % 2 == 1 else colors.white
        background_styles.append(('BACKGROUND', (0, i), (-1, i), bg_color))

    # Áp dụng toàn bộ style
    table.setStyle(TableStyle([
                                  ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                                  ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                                  ('FONTNAME', (0, 0), (-1, 0), 'Roboto_Black'),  # Header
                                  ('FONTNAME', (0, 1), (0, -1), 'Roboto_Regular'),  # Tên chỉ tiêu
                                  ('FONTNAME', (1, 1), (-1, -1), 'Roboto_Regular'),  # Dữ liệu
                                  ('FONTSIZE', (0, 0), (-1, -1), 10),
                                  ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                  ('TOPPADDING', (0, 0), (-1, -1), 6),
                              ] + background_styles))

    # Vẽ bảng chính giữa
    table.wrapOn(c, width, height)
    x_pos = (width - sum(colWidths)) / 2
    table.drawOn(c, x_pos, height - 305)

    c.setFont("Roboto_Regular", 10)
    c.drawString(165, height - 130, "(Đơn vị: triệu VND)")

    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 310, width - 40, height - 310)

    # Tạo nhận xét
    comment_text = generate_financial_commentary(bcdkt_df, stock_code)
    style_comment = ParagraphStyle(
        'Comment',
        fontName="Roboto_Regular",
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY
    )
    p = Paragraph(f"<b>Nhận xét:</b><br/>{comment_text}", style_comment)
    w, h = p.wrap(width - 80, height - 320)
    p.drawOn(c, 40, height - 320 - h)

    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 390, width - 40, height - 390)

    # Tiêu đề "Bảng kết quả kinh doanh"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 410, "Bảng kết quả kinh doanh")

    c.setFont("Roboto_Regular", 10)
    c.drawString(185, height - 410, "(Đơn vị: triệu VND)")

    # Lọc theo mã
    kqkd_stock = kqkd_df[kqkd_df['Mã'] == stock_code]

    # Lấy các năm theo thứ tự
    years = sorted(kqkd_stock['Năm'].dropna().astype(int).unique())

    # Tiêu đề 1
    headers = ["Chỉ tiêu"] + [str(y) for y in years]

    # Các chỉ tiêu cần hiển thị
    fields = [
        "Doanh thu thuần",
        "Lợi nhuận thuần từ hoạt động kinh doanh",
        "Tổng lợi nhuận kế toán trước thuế",
        "Lợi nhuận sau thuế thu nhập doanh nghiệp",
        "Lãi trước thuế"
    ]

    # Tạo dữ liệu bảng
    data = [headers]
    for field in fields:
        # Nếu cần đổi tên hiển thị
        display_name = (
            "Lợi nhuận sau thuế" if field == "Lợi nhuận sau thuế thu nhập doanh nghiệp" else field
        )
        row = [display_name]
        for y in years:
            val = kqkd_stock.loc[(kqkd_stock['Năm'] == y), field]
            if not val.empty and pd.notna(val.values[0]):
                value = int(val.values[0]) // 1_000_000  # Chuyển về triệu nếu cần
                row.append(f"{value:,}")
            else:
                row.append("Không có")
        data.append(row)

    usable_width = width - 80
    colWidths = [250] + [(usable_width - 250) / len(years)] * len(years)

    table = Table(data, colWidths=colWidths)

    # Màu nền xen kẽ
    background_styles = [('BACKGROUND', (0, 0), (-1, 0), colors.white)]
    for i in range(1, len(data)):
        bg = LIGHT_GREEN_BG if i % 2 == 1 else colors.white
        background_styles.append(('BACKGROUND', (0, i), (-1, i), bg))

    # Style bảng
    table.setStyle(TableStyle([
                                  ('ALIGN', (1, 1), (len(years), -1), 'RIGHT'),
                                  ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                                  ('FONTNAME', (0, 0), (-1, 0), 'Roboto_Black'),
                                  ('FONTNAME', (0, 1), (0, -1), 'Roboto_Regular'),
                                  ('FONTNAME', (1, 1), (-1, -1), 'Roboto_Regular'),
                                  ('FONTSIZE', (0, 0), (-1, -1), 10),
                                  ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                  ('TOPPADDING', (0, 0), (-1, -1), 6),
                              ] + background_styles))

    table.wrapOn(c, width, height)
    table.drawOn(c, 40, height - 565)

    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 575, width - 40, height - 575)

    # Nhận xét hoạt động kinh doanh
    income_comment = generate_income_commentary(kqkd_df, stock_code)
    style_income = ParagraphStyle(
        'IncomeComment',
        fontName="Roboto_Regular",
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY
    )
    p = Paragraph(f"<b>Nhận xét:</b><br/>{income_comment}", style_income)
    w, h = p.wrap(width - 80, height)
    p.drawOn(c, 40, height - 585 - h)

    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 655, width - 40, height - 655)

    # Ngắt sang trang mới
    add_page_footer(c, width)
    c.showPage()

    # Tiêu đề
    add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH)

    # Tiêu đề "Bảng lưu chuyển tiền tệ"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 107, "Bảng lưu chuyển tiền tệ")
    
    c.setFont("Roboto_Regular", 10)
    c.setFillColor(GREEN_TEXT)
    c.drawString(180, height - 107, "(Đơn vị: triệu VND)")

    # Lọc theo mã
    lctt_stock = lctt_df[lctt_df['Mã'] == stock_code]
    years = sorted(lctt_stock['Năm'].dropna().astype(int).unique())

    # Header bảng
    headers = ["Chỉ tiêu"] + [str(y) for y in years]

    # Các chỉ tiêu cần lấy và tên hiển thị
    field_map = {
        "Cổ tức đã trả (TT)": "Cổ tức đã trả",
        "Lưu chuyển tiền thuần trong kỳ (TT)": "Lưu chuyển tiền thuần",
        "Tiền và tương đương tiền đầu kỳ (TT)": "Tiền và tương đương tiền đầu kỳ",
        "Tiền và tương đương tiền cuối kỳ (TT)": "Tiền và tương đương tiền cuối kỳ"
    }

    # Tạo bảng dữ liệu
    data = [headers]
    for field, label in field_map.items():
        row = [label]
        for y in years:
            val = lctt_stock.loc[(lctt_stock['Năm'] == y), field]
            if not val.empty and pd.notna(val.values[0]):
                value = int(val.values[0]) // 1_000_000
                row.append(f"{value:,}")
            else:
                row.append("Không có")
        data.append(row)

    usable_width = width - 80
    colWidths = [250] + [(usable_width - 250) / len(years)] * len(years)

    table = Table(data, colWidths=colWidths)

    # Màu nền xen kẽ
    background_styles = [('BACKGROUND', (0, 0), (-1, 0), colors.white)]
    for i in range(1, len(data)):
        bg = LIGHT_GREEN_BG if i % 2 == 1 else colors.white
        background_styles.append(('BACKGROUND', (0, i), (-1, i), bg))

    # Style bảng
    table.setStyle(TableStyle([
                                  ('ALIGN', (1, 1), (len(years), -1), 'CENTER'),
                                  ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                                  ('FONTNAME', (0, 0), (-1, 0), 'Roboto_Black'),
                                  ('FONTNAME', (0, 1), (0, -1), 'Roboto_Regular'),
                                  ('FONTNAME', (1, 1), (-1, -1), 'Roboto_Regular'),
                                  ('FONTSIZE', (0, 0), (-1, -1), 10),
                                  ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                  ('TOPPADDING', (0, 0), (-1, -1), 6),
                              ] + background_styles))

    table.wrapOn(c, width, height)
    table.drawOn(c, 40, height - 232)

    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 242, width - 40, height - 242)

    comment = generate_cashflow_commentary(lctt_df, stock_code)
    style = ParagraphStyle(
        'CashflowComment',
        fontName="Roboto_Regular",
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY
    )
    p = Paragraph(f"<b>Nhận xét:</b><br/>{comment}", style)
    w, h = p.wrap(width - 80, height)
    p.drawOn(c, 40, height - 252 - h)

    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 322, width - 40, height - 322)
    
    # Tiêu đề "Tỷ trọng tài sản và nguồn vốn"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 342, "Tỷ trọng tài sản và nguồn vốn")
    
    c.setFont("Roboto_Regular", 10)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 362, "(Đơn vị: triệu VND)")

    # Biểu đồ tài sản & nguồn vốn
    chart_buffer = draw_asset_liability_chart(bcdkt_df, stock_code)
    if chart_buffer:
        chart_image = Image(chart_buffer, width=480, height=240)
        chart_image.hAlign = 'CENTER'
        chart_image.drawOn(c, (width - 480)/2, height - 610)
    
    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 625, width - 40, height - 625)

    #Nhận xét
    comment_text = generate_asset_liability_commentary(bcdkt_df, stock_code)
    style_comment = ParagraphStyle(
        'Comment',
        fontName="Roboto_Regular",
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY
    )
    p = Paragraph(f"<b>Nhận xét:</b><br/>{comment_text}", style_comment)
    w, h = p.wrap(width - 80, height)
    p.drawOn(c, 40, height - 635 - h)

    # Đường kẻ xanh
    c.setStrokeColor(LIGHT_GREEN)
    c.setLineWidth(1.5)
    c.line(40, height - 700, width - 40, height - 700)

    # Ngắt sang trang mới
    add_page_footer(c, width)
    c.showPage()

    # Tiêu đề
    add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH)

    # Tiêu đề "Các chỉ số tài chính"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 107, "CÁC CHỈ SỐ TÀI CHÍNH")

    # Tiêu đề "Chỉ số sinh lời (Profitability Ratios)"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 127, "1. Chỉ số sinh lời (Profitability Ratios)")

    # Lấy dữ liệu cho ROA, ROE, ROS năm 2024
    company_row = ratio_df[(ratio_df['Mã'] == stock_code) & (ratio_df['Năm'] == 2024)]
    if not company_row.empty:
        industry_name = company_row.iloc[0]["Ngành ICB - cấp 3"]
        industry_avg_df = pd.read_excel(os.path.join(DATA_PATH, "industry_avg.xlsx"))
        industry_row = industry_avg_df[
            (industry_avg_df['Ngành ICB - cấp 3'] == industry_name) & (industry_avg_df['Năm'] == 2024)]

        if not industry_row.empty:
            def extract(val):
                return round(val, 2) if pd.notna(val) else "NA"

            def compare(val1, val2):
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    return "↑" if val1 > val2 else ("↓" if val1 < val2 else "=")
                return "-"

            roe_c = extract(company_row["ROE (%)"].values[0])
            roa_c = extract(company_row["ROA (%)"].values[0])
            ros_c = extract(company_row["ROS (%)"].values[0])

            roe_i = extract(industry_row["ROE (%)"].values[0])
            roa_i = extract(industry_row["ROA (%)"].values[0])
            ros_i = extract(industry_row["ROS (%)"].values[0])

            def compare_icon_img(val1, val2):
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 > val2:
                        return Image(IMG_PATH1, width=10, height=10)
                    elif val1 < val2:
                        return Image(IMG_PATH, width=10, height=10)
                return "-"

            data = [
                ["Chỉ số", ten_cong_ty, "Trung bình ngành", "So sánh"],
                ["ROE (%)", roe_c, roe_i, compare_icon_img(roe_c, roe_i)],
                ["ROA (%)", roa_c, roa_i, compare_icon_img(roa_c, roa_i)],
                ["ROS (%)", ros_c, ros_i, compare_icon_img(ros_c, ros_i)],
            ]

            table1 = Table(data, colWidths=[130, 130, 130, 125])

            # Style nền xen kẽ cho bảng ROA/ROE/ROS
            background_styles = [('BACKGROUND', (0, 0), (-1, 0), colors.white)]

            for i in range(1, len(data)):
                bg_color = LIGHT_GREEN_BG if i % 2 == 1 else colors.white
                background_styles.append(('BACKGROUND', (0, i), (-1, i), bg_color))

            # Áp dụng style
            table1.setStyle(TableStyle([
                                           # Căn lề
                                           ('ALIGN', (1, 1), (2, -1), 'LEFT'),  # Cột Công ty & TB Ngành
                                           ('ALIGN', (3, 1), (3, -1), 'LEFT'),  # Cột So sánh
                                           ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Cột Chỉ số
                                           ('ALIGN', (0, 0), (-1, 0), 'LEFT'),  # Header

                                           ('FONTNAME', (0, 0), (-1, 0), 'Roboto_Black'),
                                           ('FONTNAME', (0, 1), (0, -1), 'Roboto_Regular'),
                                           ('FONTNAME', (1, 1), (2, -1), 'Roboto_Regular'),
                                           ('FONTSIZE', (0, 0), (-1, -1), 10),
                                           ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                           ('TOPPADDING', (0, 0), (-1, -1), 6),
                                       ] + background_styles))

            # Vẽ bảng
            table1.wrapOn(c, width, height)
            table1.drawOn(c, 40, height - 230)

    # Tiêu đề "Biểu đồ so sánh"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 260, "Biểu đồ so sánh")

    # Gọi hàm tạo biểu đồ
    chart_buffer = draw_profitability_chart(ratio_df, stock_code)

    # Chèn vào PDF
    if chart_buffer:
        c.setFont("Roboto_Black", 14)
        c.setFillColor(GREEN_TEXT)

        chart_img = Image(chart_buffer, width=520, height=310)
        chart_img.wrapOn(c, width, height)
        chart_img.drawOn(c, 40, height - 575)

    # Tiêu đề "Nhận xét"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 585, "Nhận xét")

    # Nhận xét tự động chi tiết
    if not company_row.empty and not industry_row.empty:
        roa_diff = roa_c - roa_i
        roe_diff = roe_c - roe_i
        ros_diff = ros_c - ros_i

        comment = "Các chỉ số sinh lời (ROA, ROE, ROS) phản ánh hiệu quả hoạt động của doanh nghiệp trong việc sử dụng tài sản, vốn chủ sở hữu và khả năng sinh lợi từ doanh thu. Dưới đây là phần phân tích chi tiết:<br/>"

        # ROA
        if isinstance(roa_diff, (int, float)):
            if roa_diff > 0:
                comment += f"- ROA (Tỷ suất lợi nhuận trên tổng tài sản) của công ty đạt {roa_c}%, cao hơn mức trung bình ngành là {roa_i}%. Điều này cho thấy công ty đang sử dụng tài sản một cách hiệu quả để tạo ra lợi nhuận. Đây là dấu hiệu tích cực, thể hiện năng lực vận hành ổn định và có thể là kết quả của việc tối ưu chi phí hoạt động hoặc cấu trúc tài sản hợp lý.<br/>"
            elif roa_diff < 0:
                comment += f"- ROA của công ty chỉ đạt {roa_c}%, thấp hơn trung bình ngành là {roa_i}%. Điều này cho thấy hiệu quả sử dụng tài sản chưa tối ưu. Công ty có thể cần đánh giá lại cơ cấu tài sản, hoặc xem xét lại hoạt động vận hành để nâng cao hiệu quả sử dụng nguồn lực hiện có.<br/>"
            else:
                comment += "- ROA của công ty tương đương với trung bình ngành, cho thấy hiệu quả sử dụng tài sản ở mức trung bình so với đối thủ cạnh tranh.<br/>"

        # ROE
        if isinstance(roe_diff, (int, float)):
            if roe_diff > 0:
                comment += f"- ROE (Tỷ suất lợi nhuận trên vốn chủ sở hữu) của công ty đạt {roe_c}%, vượt trung bình ngành ({roe_i}%). Điều này chứng tỏ công ty có khả năng tạo ra giá trị cao cho cổ đông từ vốn đầu tư. Đây là một điểm mạnh cần duy trì và phát huy, đặc biệt trong việc thu hút nhà đầu tư.<br/>"
            elif roe_diff < 0:
                comment += f"- ROE của công ty là {roe_c}%, thấp hơn mức trung bình ngành là {roe_i}%. Điều này phản ánh khả năng tạo lợi nhuận từ vốn chủ sở hữu chưa hiệu quả, có thể do lợi nhuận ròng thấp hoặc vốn đầu tư chưa được khai thác đúng cách. Doanh nghiệp nên xem xét lại chiến lược sử dụng vốn hoặc cơ cấu tài chính.<br/>"
            else:
                comment += "- ROE của công ty tương đương trung bình ngành, thể hiện hiệu suất sinh lời trên vốn đầu tư ở mức phổ biến trong ngành.<br/>"

        # ROS
        if isinstance(ros_diff, (int, float)):
            if ros_diff > 0:
                comment += f"- ROS (Tỷ suất lợi nhuận trên doanh thu) của công ty là {ros_c}%, cao hơn trung bình ngành ({ros_i}%). Điều này thể hiện khả năng kiểm soát chi phí tốt và tạo ra lợi nhuận cao từ doanh thu thuần. Đây là một lợi thế cạnh tranh trong ngành có biên lợi nhuận thấp.<br/>"
            elif ros_diff < 0:
                comment += f"- ROS chỉ đạt {ros_c}%, thấp hơn trung bình ngành ({ros_i}%). Điều này có thể cho thấy công ty đang đối mặt với áp lực chi phí cao hoặc không tận dụng được lợi thế về giá bán. Cần đánh giá lại chiến lược chi phí, định giá và cấu trúc sản phẩm.<br/>"
            else:
                comment += "- ROS của công ty ngang bằng với trung bình ngành, cho thấy biên lợi nhuận ròng ở mức trung bình so với các đối thủ.<br/>"

        comment += "Tóm lại, việc so sánh các chỉ số sinh lời với trung bình ngành giúp đánh giá vị thế cạnh tranh của doanh nghiệp. Nếu các chỉ số cao hơn, công ty có lợi thế về hiệu quả và năng lực sinh lời. Ngược lại, nếu thấp hơn, cần xem xét chiến lược quản trị tài sản, chi phí và vốn để cải thiện hiệu quả hoạt động."

        # Hiển thị đoạn nhận xét lên PDF
        style_comment = ParagraphStyle(
            name="Justify",
            fontName="Roboto_Regular",
            fontSize=11,
            leading=15,
            alignment=TA_JUSTIFY,
        )

        comment_paragraph = Paragraph(comment, style_comment)
        w, h_comment = comment_paragraph.wrap(width - 80, height)
        comment_paragraph.drawOn(c, 40, height - 595 - h_comment)

    # Ngắt sang trang mới
    add_page_footer(c, width)
    c.showPage()

    # Tiêu đề
    add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH)

    # Tiêu đề "Các chỉ số tài chính"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 107, "CÁC CHỈ SỐ TÀI CHÍNH")

    # Tiêu đề "Chỉ số định giá (Valuation Ratios)"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 127, "2. Chỉ số định giá (Valuation Ratios)")

    # Đọc dữ liệu ngành
    industry_avg_df = pd.read_excel(os.path.join(DATA_PATH, "industry_avg.xlsx"))
    industry_row = industry_avg_df[
        (industry_avg_df['Ngành ICB - cấp 3'] == nganh_cap3) & (industry_avg_df['Năm'] == selected_date.year)
        ]

    # Lấy P/E và P/B của công ty
    if not ratio_row.empty:
        pe_value = ratio_row.iloc[0].get("P/E", None)
    else:
        pe_value = None
        print(f"⚠️ Không có dữ liệu P/E cho {stock_code} năm {selected_date.year}")
    if not ratio_row.empty:
        pb_value = ratio_row.iloc[0].get("P/B", None)
    else:
        pb_value = None
        print(f"⚠️ Không có dữ liệu P/B cho {stock_code} năm {selected_date.year}")

    # Lấy P/E và P/B của ngành
    pe_ind = industry_row["P/E"].values[0] if not industry_row.empty else "NA"
    pb_ind = industry_row["P/B"].values[0] if not industry_row.empty else "NA"

    # So sánh bằng biểu tượng
    def compare_icon(val1, val2):
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if val1 > val2:
                return Image(IMG_PATH1, width=10, height=10)
            elif val1 < val2:
                return Image(IMG_PATH, width=10, height=10)
        return "-"

    # Tạo bảng dữ liệu
    valuation_data = [
        ["Chỉ số", ten_cong_ty, "Trung bình ngành", "So sánh"],
        ["P/E", f"{round(pe_value, 2):,}" if isinstance(pe_value, (int, float)) else "NA",
         f"{round(pe_ind, 2):,}" if isinstance(pe_ind, (int, float)) else "NA",
         compare_icon(pe_value, pe_ind)],
        ["P/B", f"{round(pb_value, 2):,}" if isinstance(pb_value, (int, float)) else "NA",
         f"{round(pb_ind, 2):,}" if isinstance(pb_ind, (int, float)) else "NA",
         compare_icon(pb_value, pb_ind)],
    ]

    valuation_table = Table(valuation_data, colWidths=[130, 130, 130, 125])

    # Màu nền xen kẽ
    valuation_styles = [('BACKGROUND', (0, 0), (-1, 0), colors.white)]
    for i in range(1, len(valuation_data)):
        bg_color = LIGHT_GREEN_BG if i % 2 == 1 else colors.white
        valuation_styles.append(('BACKGROUND', (0, i), (-1, i), bg_color))

    # Áp dụng TableStyle
    valuation_table.setStyle(TableStyle([
                                            ('ALIGN', (1, 1), (2, -1), 'LEFT'),
                                            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                                            ('ALIGN', (3, 1), (3, -1), 'LEFT'),
                                            ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
                                            ('FONTNAME', (0, 0), (-1, 0), 'Roboto_Black'),
                                            ('FONTNAME', (0, 1), (-1, -1), 'Roboto_Regular'),
                                            ('FONTSIZE', (0, 0), (-1, -1), 10),
                                            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                            ('TOPPADDING', (0, 0), (-1, -1), 6),
                                        ] + valuation_styles))

    # Vẽ bảng
    valuation_table.wrapOn(c, width, height)
    valuation_table.drawOn(c, 40, height - 210)

    # Tiêu đề "Biểu đồ so sánh"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 230, "Biểu đồ so sánh")

    # Vẽ biểu đồ
    chart_buffer = draw_valuation_chart(ratio_df, pd.read_excel(os.path.join(DATA_PATH, "industry_avg.xlsx")), stock_code)
    if chart_buffer:
        chart_img = Image(chart_buffer, width=500, height=300)
        chart_img.wrapOn(c, width, height)
        chart_img.drawOn(c, 40, height - 545)

    # Tiêu đề "Nhận xét"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 565, "Nhận xét")

    # ----- NHẬN XÉT P/E VÀ P/B -----
    pe_diff = pe_value - pe_ind if isinstance(pe_value, (int, float)) and isinstance(pe_ind, (int, float)) else None
    pb_diff = pb_value - pb_ind if isinstance(pb_value, (int, float)) and isinstance(pb_ind, (int, float)) else None

    valuation_note = ""

    if isinstance(pe_value, (int, float)) and isinstance(pe_ind, (int, float)):
        pe_diff = pe_value - pe_ind
        if pe_diff > 0:
            valuation_note += f"- Chỉ số P/E của công ty đang cao hơn trung bình ngành khoảng {pe_diff:.2f} lần. Điều này có thể cho thấy nhà đầu tư đang kỳ vọng vào tiềm năng tăng trưởng trong tương lai, hoặc cổ phiếu đang bị định giá cao so với lợi nhuận hiện tại.\n"
        elif pe_diff < 0:
            valuation_note += f"- Chỉ số P/E thấp hơn mức trung bình ngành khoảng {abs(pe_diff):.2f} lần. Đây có thể là dấu hiệu của mức giá hợp lý hoặc do thị trường đánh giá thấp khả năng sinh lời trong tương lai của công ty.\n"
        else:
            valuation_note += "- Chỉ số P/E của công ty gần như tương đương với trung bình ngành, phản ánh mức định giá ổn định theo mặt bằng chung.\n"
    else:
        valuation_note += "- Không có đủ dữ liệu để đánh giá chỉ số P/E của công ty so với ngành.\n"

    if isinstance(pb_value, (int, float)) and isinstance(pb_ind, (int, float)):
        pb_diff = pb_value - pb_ind
        if pb_diff > 0:
            valuation_note += f"- Chỉ số P/B cao hơn trung bình ngành khoảng {pb_diff:.2f} lần, cho thấy thị trường có thể đang đánh giá cao tài sản vô hình hoặc khả năng sinh lợi trong tương lai của doanh nghiệp.\n"
        elif pb_diff < 0:
            valuation_note += f"- Chỉ số P/B thấp hơn trung bình ngành khoảng {abs(pb_diff):.2f} lần, điều này có thể phản ánh sự dè dặt của thị trường hoặc dấu hiệu tiềm ẩn về hiệu quả sử dụng tài sản.\n"
        else:
            valuation_note += "- Chỉ số P/B của công ty xấp xỉ mức trung bình ngành.\n"
    else:
        valuation_note += "- Không có đủ dữ liệu để so sánh chỉ số P/B.\n"

    # Vẽ đoạn nhận xét ra PDF
    styleN = ParagraphStyle(
        'Normal',
        fontName="Roboto_Regular",
        fontSize=11,
        leading=15,
        alignment=TA_JUSTIFY,
    )

    p = Paragraph(valuation_note.replace("\n", "<br/>"), styleN)
    w, h = p.wrap(width - 80, height)
    p.drawOn(c, 40, height - 575 - h)

    # Ngắt sang trang mới
    add_page_footer(c, width)
    c.showPage()

    # Tiêu đề
    add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH)

    # Tiêu đề "Các chỉ số tài chính"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 107, "CÁC CHỈ SỐ TÀI CHÍNH")

    # Tiêu đề "3. Chỉ số tăng trưởng (Growth Ratios)"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 127, "3. Chỉ số tăng trưởng (Growth Ratios)")

    # Lấy dữ liệu từ industry_avg
    growth_company = ratio_df[(ratio_df['Mã'] == stock_code) & (ratio_df['Năm'] == 2024)]
    industry_row = industry_avg_df[
        (industry_avg_df['Ngành ICB - cấp 3'] == industry_name) & (industry_avg_df['Năm'] == 2024)
        ]

    if not growth_company.empty and not industry_row.empty:
        def extract(val):
            return round(val, 2) if pd.notna(val) else "NA"

        rev_growth_c = extract(growth_company["Revenue Growth (%)"].values[0])
        net_growth_c = extract(growth_company["Net Income Growth (%)"].values[0])

        rev_growth_i = extract(industry_row["Revenue Growth (%)"].values[0])
        net_growth_i = extract(industry_row["Net Income Growth (%)"].values[0])

        def compare_icon(val1, val2):
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 > val2:
                    return Image(IMG_PATH1, width=10, height=10)
                elif val1 < val2:
                    return Image(IMG_PATH, width=10, height=10)
            return "-"

        data_growth = [
            ["Chỉ số", ten_cong_ty, "Trung bình ngành", "So sánh"],
            ["Revenue Growth (%)", rev_growth_c, rev_growth_i, compare_icon(rev_growth_c, rev_growth_i)],
            ["Net Income Growth (%)", net_growth_c, net_growth_i, compare_icon(net_growth_c, net_growth_i)],
        ]

        table_growth = Table(data_growth, colWidths=[150, 130, 130, 125])

        background_styles = [('BACKGROUND', (0, 0), (-1, 0), colors.white)]
        for i in range(1, len(data_growth)):
            bg = LIGHT_GREEN_BG if i % 2 == 1 else colors.white
            background_styles.append(('BACKGROUND', (0, i), (-1, i), bg))

        table_growth.setStyle(TableStyle([
                                             ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
                                             ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                                             ('ALIGN', (1, 1), (2, -1), 'LEFT'),
                                             ('ALIGN', (3, 1), (3, -1), 'LEFT'),

                                             ('FONTNAME', (0, 0), (-1, 0), 'Roboto_Black'),
                                             ('FONTNAME', (0, 1), (0, -1), 'Roboto_Regular'),
                                             ('FONTNAME', (1, 1), (2, -1), 'Roboto_Regular'),
                                             ('FONTSIZE', (0, 0), (-1, -1), 10),
                                             ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                             ('TOPPADDING', (0, 0), (-1, -1), 6),
                                         ] + background_styles))

        # Vẽ bảng vào PDF
        table_growth.wrapOn(c, width, height)
        table_growth.drawOn(c, 40, height - 210)

    # Tiêu đề "Biểu đồ so sánh"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 230, "Biểu đồ so sánh")

    #Vẽ chart
    chart_buffer = draw_growth_chart(ratio_df, stock_code)
    if chart_buffer:
        img = Image(chart_buffer, width=520, height=300)
        img.wrapOn(c, width, height)
        img.drawOn(c, 40, height - 540)

    # Tiêu đề "Nhận xét"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 550, "Nhận xét")

    def generate_growth_comment(ratio_df, stock_code):
        df_plot = ratio_df[(ratio_df["Mã"] == stock_code) & (ratio_df["Năm"].between(2020, 2024))].sort_values("Năm")
        if df_plot.empty:
            return "Không đủ dữ liệu để đưa ra nhận xét cụ thể về xu hướng tăng trưởng doanh thu và lợi nhuận trong giai đoạn 2020–2024."

        rev_growth = df_plot["Revenue Growth (%)"].dropna()
        net_growth = df_plot["Net Income Growth (%)"].dropna()

        avg_rev = rev_growth.mean() if not rev_growth.empty else None
        avg_net = net_growth.mean() if not net_growth.empty else None

        comment = "Phân tích xu hướng tăng trưởng doanh thu và lợi nhuận sau thuế của công ty trong giai đoạn 2020–2024 cho thấy:\n"

        # Doanh thu
        if avg_rev is not None:
            if avg_rev > 10:
                comment += (
                    f"- Doanh thu có tốc độ tăng trưởng ấn tượng, với mức trung bình hàng năm đạt khoảng {avg_rev:.2f}%. "
                    "Điều này phản ánh khả năng mở rộng thị trường, phát triển sản phẩm hoặc dịch vụ mới hiệu quả của doanh nghiệp. "
                    "Một xu hướng như vậy thường là tín hiệu tích cực đối với các nhà đầu tư, bởi nó cho thấy công ty có nền tảng tăng trưởng bền vững trong dài hạn.\n"
                )
            elif avg_rev > 0:
                comment += (
                    f"- Doanh thu ghi nhận mức tăng trưởng trung bình {avg_rev:.2f}% mỗi năm. "
                    "Dù không thực sự bứt phá, đây vẫn là dấu hiệu cho thấy công ty duy trì được đà tăng trưởng ổn định, "
                    "mặc dù tốc độ này có thể chưa đủ mạnh để tạo ra lợi thế cạnh tranh rõ nét trên thị trường.\n"
                )
            else:
                comment += (
                    f"- Doanh thu có chiều hướng giảm nhẹ, với mức trung bình khoảng {avg_rev:.2f}%. "
                    "Sự sụt giảm này có thể phản ánh những khó khăn trong việc duy trì thị phần, hoặc ảnh hưởng từ yếu tố bên ngoài như điều kiện kinh tế vĩ mô. "
                    "Nếu xu hướng này tiếp tục kéo dài, công ty cần nhanh chóng đánh giá lại chiến lược kinh doanh để tránh rơi vào tình trạng suy giảm kéo dài.\n"
                )
        else:
            comment += "- Không có đủ dữ liệu để đánh giá xu hướng tăng trưởng doanh thu trong giai đoạn này.\n"

        # Lợi nhuận sau thuế
        if avg_net is not None:
            if avg_net > 10:
                comment += (
                    f"- Lợi nhuận sau thuế tăng trưởng mạnh mẽ, trung bình đạt khoảng {avg_net:.2f}% mỗi năm. "
                    "Đây là dấu hiệu rõ ràng cho thấy công ty không chỉ tăng doanh thu mà còn kiểm soát tốt chi phí, cải thiện hiệu quả hoạt động. "
                    "Tăng trưởng lợi nhuận như vậy góp phần củng cố niềm tin của nhà đầu tư vào triển vọng dài hạn của doanh nghiệp.\n"
                )
            elif avg_net > 0:
                comment += (
                    f"- Lợi nhuận sau thuế tăng trưởng với mức trung bình {avg_net:.2f}%. "
                    "Dù chưa thực sự bứt phá, nhưng vẫn cho thấy công ty đang đi đúng hướng trong việc nâng cao hiệu quả kinh doanh. "
                    "Tuy nhiên, công ty cần tiếp tục tối ưu hóa biên lợi nhuận để chuyển đổi tăng trưởng doanh thu thành lợi nhuận tốt hơn.\n"
                )
            else:
                comment += (
                    f"- Lợi nhuận sau thuế có dấu hiệu suy giảm, với mức trung bình {avg_net:.2f}%. "
                    "Đây có thể là hệ quả từ việc chi phí vận hành tăng nhanh hơn doanh thu, hoặc những yếu tố bất lợi như chi phí tài chính, thuế, hay biến động thị trường. "
                    "Việc lợi nhuận sụt giảm là tín hiệu cần được theo dõi chặt chẽ vì có thể ảnh hưởng đến khả năng sinh lời và phân phối cổ tức trong tương lai.\n"
                )
        else:
            comment += "- Không có đủ dữ liệu để đánh giá xu hướng tăng trưởng lợi nhuận sau thuế trong giai đoạn này.\n"

        return comment.strip()

    comment = generate_growth_comment(ratio_df, stock_code)

    style = ParagraphStyle(
        name="GrowthComment",
        fontName="Roboto_Regular",
        fontSize=11,
        leading=15,
        alignment=TA_JUSTIFY,
    )
    para = Paragraph(comment.replace("\n", "<br/>"), style)
    w, h = para.wrap(width - 80, height)
    para.drawOn(c, 40, height - 690)

    # Ngắt sang trang mới
    add_page_footer(c, width)
    c.showPage()

    # Tiêu đề
    add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH)

    # Tiêu đề "Các chỉ số tài chính"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 107, "CÁC CHỈ SỐ TÀI CHÍNH")

    # Tiêu đề "4. Chỉ số đòn bẩy tài chính (Leverage Ratios)"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 127, "4. Chỉ số đòn bẩy tài chính (Leverage Ratios)")

    # Lấy dữ liệu
    da_c = extract(company_row["D/A (%)"].values[0])
    de_c = extract(company_row["D/E (%)"].values[0])
    ea_c = extract(company_row["E/A (%)"].values[0])

    da_i = extract(industry_row["D/A (%)"].values[0])
    de_i = extract(industry_row["D/E (%)"].values[0])
    ea_i = extract(industry_row["E/A (%)"].values[0])

    def compare_icon(val1, val2):
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if val1 > val2:
                return Image(IMG_PATH1, width=10, height=10)
            elif val1 < val2:
                return Image(IMG_PATH, width=10, height=10)
        return "-"

    data_leverage = [
        ["Chỉ số", ten_cong_ty, "Trung bình ngành", "So sánh"],
        ["D/A (%)", da_c, da_i, compare_icon(da_c, da_i)],
        ["D/E (%)", de_c, de_i, compare_icon(de_c, de_i)],
        ["E/A (%)", ea_c, ea_i, compare_icon(ea_c, ea_i)],
    ]

    # Nền xen kẽ
    background_styles = [('BACKGROUND', (0, 0), (-1, 0), colors.white)]
    for i in range(1, len(data_leverage)):
        bg = LIGHT_GREEN_BG if i % 2 == 1 else colors.white
        background_styles.append(('BACKGROUND', (0, i), (-1, i), bg))

    # Tạo bảng
    table = Table(data_leverage, colWidths=[130, 130, 130, 125])
    table.setStyle(TableStyle([
                                  ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
                                  ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                                  ('ALIGN', (1, 1), (2, -1), 'LEFT'),
                                  ('ALIGN', (3, 1), (3, -1), 'LEFT'),
                                  ('FONTNAME', (0, 0), (-1, 0), 'Roboto_Black'),
                                  ('FONTNAME', (0, 1), (-1, -1), 'Roboto_Regular'),
                                  ('FONTSIZE', (0, 0), (-1, -1), 10),
                                  ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                  ('TOPPADDING', (0, 0), (-1, -1), 6),
                              ] + background_styles))

    table.wrapOn(c, width, height)
    table.drawOn(c, 40, height - 230)

    # Tiêu đề "Biểu đồ so sánh"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 250, "Biểu đồ so sánh")

    chart_buffer = draw_leverage_chart(ratio_df, stock_code)
    if chart_buffer:
        leverage_chart = Image(chart_buffer, width=520, height=300)
        leverage_chart.wrapOn(c, width, height)
        leverage_chart.drawOn(c, 40, height - 570)

    # Tiêu đề "Nhận xét"
    c.setFont("Roboto_Black", 14)
    c.drawString(40, height - 620, "Nhận xét")

    def generate_leverage_comment(ratio_df, stock_code):
        df_plot = ratio_df[(ratio_df["Mã"] == stock_code) & (ratio_df["Năm"].between(2020, 2024))].sort_values("Năm")
        if df_plot.empty:
            return "Không đủ dữ liệu để nhận xét về các chỉ số đòn bẩy tài chính."

        da_series = df_plot["D/A (%)"].dropna()
        de_series = df_plot["D/E (%)"].dropna()
        ea_series = df_plot["E/A (%)"].dropna()

        avg_da = da_series.mean() if not da_series.empty else None
        avg_de = de_series.mean() if not de_series.empty else None
        avg_ea = ea_series.mean() if not ea_series.empty else None

        comment = "Phân tích các chỉ số đòn bẩy tài chính giai đoạn 2020–2024:\n"

        # Nhận xét D/A (%)
        if avg_da is not None:
            comment += f"- Tỷ lệ D/A trung bình khoảng {avg_da:.2f}%, phản ánh tỷ trọng nợ trong tổng tài sản của công ty "
            if avg_da > 50:
                comment += "ở mức cao, cho thấy công ty đang dựa nhiều vào nợ để tài trợ hoạt động, tiềm ẩn rủi ro tài chính nếu thị trường biến động mạnh.\n"
            elif avg_da >= 30:
                comment += "ở mức tương đối, cho thấy công ty đang cân bằng giữa vốn chủ sở hữu và nợ vay.\n"
            else:
                comment += "ở mức thấp, cho thấy công ty chủ yếu tài trợ bằng vốn chủ sở hữu, ít phụ thuộc vào nợ.\n"
        else:
            comment += "- Không đủ dữ liệu để đánh giá tỷ lệ D/A.\n"

        # Nhận xét D/E (%)
        if avg_de is not None:
            comment += f"- Tỷ lệ D/E trung bình khoảng {avg_de:.2f}%, phản ánh mức độ đòn bẩy tài chính của công ty. "
            if avg_de > 150:
                comment += "Tỷ lệ này khá cao, cho thấy công ty có thể gặp áp lực trả nợ lớn.\n"
            elif avg_de >= 80:
                comment += "Tỷ lệ này ở mức chấp nhận được, thể hiện công ty có sử dụng đòn bẩy nhưng chưa vượt ngưỡng rủi ro.\n"
            else:
                comment += "Tỷ lệ khá thấp, cho thấy công ty thận trọng trong vay nợ và ưu tiên vốn chủ sở hữu.\n"
        else:
            comment += "- Không đủ dữ liệu để đánh giá tỷ lệ D/E.\n"

        # Nhận xét E/A (%)
        if avg_ea is not None:
            comment += f"- Tỷ lệ E/A trung bình là {avg_ea:.2f}%, thể hiện tỷ trọng vốn chủ sở hữu trong tổng tài sản. "
            if avg_ea > 60:
                comment += "Tỷ lệ này cao cho thấy công ty có nền tảng tài chính vững chắc, ít phụ thuộc vào nợ.\n"
            elif avg_ea >= 40:
                comment += "Tỷ lệ ổn định, phản ánh sự cân đối trong cấu trúc vốn.\n"
            else:
                comment += "Tỷ lệ thấp, có thể khiến công ty gặp khó khăn khi huy động vốn trong điều kiện thị trường xấu.\n"
        else:
            comment += "- Không đủ dữ liệu để đánh giá tỷ lệ E/A.\n"

        return comment.strip()

    comment = generate_leverage_comment(ratio_df, stock_code)
    style = ParagraphStyle(
        name="GrowthComment",
        fontName="Roboto_Regular",
        fontSize=11,
        leading=15,
        alignment=TA_JUSTIFY,
    )
    para = Paragraph(comment.replace("\n", "<br/>"), style)
    w, h = para.wrap(width - 80, height)
    para.drawOn(c, 40, height - 735)

    # Ngắt sang trang mới
    add_page_footer(c, width)
    c.showPage()

    # Tiêu đề
    add_page_header(c, width, height, ten_cong_ty, stock_price, ngay_tao, logo_path=LOGO_PATH)

    # Tiêu đề "Khuyến nghị dành cho nhà đầu tư"
    c.setFont("Roboto_Black", 14)
    c.setFillColor(GREEN_TEXT)
    c.drawString(40, height - 107, "KHUYẾN NGHỊ DÀNH CHO NHÀ ĐẦU TƯ")

    # Mô tả ngắn
    c.setFillColor(colors.black)
    c.setFont("Roboto_Regular", 11)
    c.drawString(40, height - 130, "Đánh giá tổng hợp và khuyến nghị đầu tư:")

    # Tạo kết quả phân tích summary (bạn dùng hàm generate_summary_data(...) ở trước đó)
    summary = generate_summary_data(ratio_df, industry_avg_df, lctt_df, stock_code)
    # Tạo nhận xét tổng kết
    recommendation_text = generate_investment_recommendation(summary)
    # Vẽ đoạn nhận xét
    style_summary = ParagraphStyle(
        name="Summary",
        fontName="Roboto_Regular",
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
    )
    # Tô xanh các tiêu đề đoạn (PHÂN TÍCH TÀI CHÍNH, ĐÁNH GIÁ TRIỂN VỌNG, KẾT LUẬN)
    recommendation_text = re.sub(r"(PHÂN TÍCH TÀI CHÍNH:|ĐÁNH GIÁ TRIỂN VỌNG:|KẾT LUẬN:)",
                                 r"<b><font color='#1B5E20'>\1</font></b>",
                                 recommendation_text)
    
    p = Paragraph(recommendation_text.replace("\n", "<br/>"), style_summary)
    p.wrapOn(c, width - 80, height)
    p.drawOn(c, 40, height - 360)

    add_page_footer(c, width)
    c.save()
    buffer.seek(0)
    return buffer, ten_cong_ty

# Giao diện Streamlit
st.title("THÔNG TIN CHỨNG KHOÁN")

stock_code = st.selectbox("Chọn mã chứng khoán", df["Mã"].unique())
selected_date = st.date_input("Chọn ngày báo cáo", min_value=min_date, max_value=max_date, value=max_date)

if st.button("📥 Tạo PDF"):
    pdf_buffer, ten_cong_ty = create_pdf(stock_code, selected_date)
    if pdf_buffer:
        file_name = f"Thông tin doanh nghiệp {ten_cong_ty}.pdf"
        st.download_button(label="📥 Tải PDF", data=pdf_buffer, file_name=file_name, mime="application/pdf")
    else:
        st.error("Không tìm thấy thông tin mã chứng khoán!")
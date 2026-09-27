import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc file Excel từ máy tính
# Tham số header=1 vì dòng đầu tiên của file này là X1, X2... Dòng thứ 2 mới là tên cột chuẩn
print("Đang tải dữ liệu...")
df = pd.read_excel('default of credit card clients.xls', header=1)

# Xóa cột ID đi vì nó chỉ là số thứ tự
df = df.drop(columns=['ID'])
# Đổi tên biến mục tiêu
df = df.rename(columns={'default payment next month': 'default'})

print(f"Kích thước tập dữ liệu: {df.shape}")

# 2. Xem thông tin tổng quan
print("\n" + "="*50)
print("--- THÔNG TIN TỔNG QUAN ---")
df.info()

# 3. Thống kê mô tả
print("\n" + "="*50)
print("--- THỐNG KÊ MÔ TẢ ---")
print(df.describe().T)

# 4. In Tỷ lệ phần trăm
print("\n" + "="*50)
print("--- TỶ LỆ TRẠNG THÁI VỠ NỢ ---")
ti_le = df['default'].value_counts(normalize=True) * 100
print(f"Tỷ lệ không vỡ nợ: {ti_le[0]:.2f}%")
print(f"Tỷ lệ vỡ nợ: {ti_le[1]:.2f}%")

# 5. Vẽ biểu đồ phân phối biến mục tiêu
print("\nĐang mở cửa sổ vẽ biểu đồ... (Vui lòng tắt cửa sổ biểu đồ để code chạy xong)")
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='default', data=df, palette='Set2')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Phân phối trạng thái vỡ nợ của khách hàng (0: Không, 1: Có)')
plt.xlabel('Trạng thái vỡ nợ (default)')
plt.ylabel('Số lượng khách hàng')

# Hiển thị biểu đồ (Lệnh này sẽ bật ra một cửa sổ mới)
plt.show()

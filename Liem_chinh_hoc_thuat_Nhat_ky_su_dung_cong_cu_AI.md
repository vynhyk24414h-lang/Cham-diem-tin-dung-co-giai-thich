# Báo cáo Liêm chính học thuật & Nhật ký sử dụng công cụ AI

**Tên dự án:** Chấm điểm tín dụng có giải thích (Explainable Credit Scoring)
**Công cụ AI được sử dụng:** Antigravity AI / Gemini 

Dưới đây là bản kê chi tiết các phần trong bài làm có sự tham khảo và hỗ trợ từ công cụ AI:

| Ngày thực hiện | Phần bài làm được AI hỗ trợ | Prompt (Câu lệnh) đã sử dụng | Chi tiết mức độ AI hỗ trợ |
| :--- | :--- | :--- | :--- |
| 27/09/2026 | **1. Tìm hiểu bối cảnh dữ liệu:** Hiểu ý nghĩa của tập dữ liệu UCI Credit Card, các biến độc lập và biến mục tiêu. | *"thẻ tín dụng UCI là gì"* | AI giải thích lý thuyết, dịch thuật thông tin về bộ dữ liệu gốc của Yeh & Lien (2009). Sinh viên tiếp thu để hiểu bài toán. |
| 27/09/2026 | **2. Cài đặt môi trường & Quản lý source code:** Tạo danh sách thư viện (`requirements.txt`) và cấu hình Git (`.gitignore`). | *"hướng dẫn tôi chi tiết từng bước... Thiết lập GitHub repo, requirements.txt..."* | AI cung cấp danh sách tên các thư viện cần thiết và logic chặn file rác trên git. Sinh viên tự gõ lệnh trên Terminal để cài đặt và khởi tạo. |
| 27/09/2026 | **3. Viết mã nguồn Khám phá dữ liệu (EDA):** Hoàn thiện toàn bộ file `01_EDA.ipynb` (đọc dữ liệu, thống kê mô tả, vẽ biểu đồ). | (Nối tiếp yêu cầu phía trên) | AI cung cấp các đoạn mã nguồn Python mẫu (dùng `pandas`, `seaborn`). Sinh viên tự copy, chạy code và đối chiếu kết quả biểu đồ. |
| 27/09/2026 | **4. Xử lý thao tác công cụ:** Hướng dẫn giải nén file thủ công, cách tạo và thao tác cơ bản trên Jupyter Notebook. | *"tôi muốn làm thủ công", "cách giải nén", "cách dán?"* | AI hướng dẫn thao tác sử dụng phần mềm VS Code cơ bản và luồng tải/giải nén file trên Windows. Sinh viên tự thao tác. |
| 27/09/2026 | **5. Sửa lỗi lập trình (Debugging):** Xử lý lỗi không đọc được file `.xls` định dạng cũ. | *(Gửi log mã lỗi ModuleNotFoundError: No module named 'xlrd')* | AI chẩn đoán lỗi và đưa ra câu lệnh cài đặt thư viện `xlrd` bổ sung. Sinh viên tự chạy lệnh khắc phục. |
| 27/09/2026 | **6. Đẩy mã nguồn lên GitHub:** Hoàn tất luồng đưa toàn bộ source code và dữ liệu lên web. | *(Gửi hình ảnh giao diện tạo repo trên GitHub)* | AI cung cấp bộ 5 câu lệnh git cơ bản (`add`, `commit`, `branch`, `remote`, `push`). Sinh viên tự thực thi lệnh. |


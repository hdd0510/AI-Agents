# Medical AI System - LangGraph

Hệ thống AI y tế đa agent sử dụng LangGraph để phân tích hình ảnh nội soi tiêu hóa, hỗ trợ phát hiện và phân loại polyp, đồng thời trả lời các câu hỏi y tế với cơ chế reflection để giảm thiểu bias.

## Tổng quan

Hệ thống được thiết kế theo kiến trúc đa agent dạng graph, trong đó mỗi agent là một node trong graph thực hiện một nhiệm vụ chuyên biệt:

- **Detector Agent**: Phát hiện polyp trong hình ảnh nội soi sử dụng YOLO
- **Classifier Agents**: Phân loại kỹ thuật chụp và vị trí giải phẫu
- **VQA Agent**: Trả lời câu hỏi dựa trên hình ảnh sử dụng LLaVA
- **Reflection Node**: Phát hiện và điều chỉnh bias trong câu trả lời

Tất cả các node được kết nối và điều phối bởi LangGraph, sử dụng các router logic để quyết định luồng xử lý dựa trên loại tác vụ.

## Kiến trúc hệ thống

Hệ thống sử dụng LangGraph để tạo và quản lý luồng công việc giữa các agent:

```
┌─────────────────────────────────────────────────────────┐
│                   LangGraph State                        │
└────────────┬────────────┬────────────┬────────────┬─────┘
             │            │            │            │
┌────────────▼─┐ ┌────────▼───────┐ ┌──▼──────┐ ┌───▼─────┐
│    Detector   │ │   Classifier   │ │   VQA   │ │Reflection│
│     Node      │ │     Nodes      │ │  Node   │ │  Node   │
└──────────────┘ └────────────────┘ └─────────┘ └─────────┘
       │                │                │            │
       └────────────────┴────────────────┴────────────┘
                               │
                     ┌─────────▼─────────┐
                     │  Result Synthesis │
                     │       Node        │
                     └───────────────────┘
```

## Cách sử dụng

### Cài đặt

```bash
# Clone repository
git clone https://github.com/yourusername/medical-ai-system.git
cd medical-ai-system

# Tạo môi trường virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt package
pip install -e .
```

### Command Line Interface

```bash
# Phân tích hình ảnh
medical-ai analyze --image path/to/image.jpg --query "Is there a polyp in this image?"

# Khởi động API service
medical-ai serve --host 0.0.0.0 --port 8000

# Hiển thị phiên bản
medical-ai version
```

### Sử dụng trong code

```python
from medical_ai_system import MedicalAISystem, MedicalGraphConfig

# Tạo config
config = MedicalGraphConfig(
    device="cuda",  # hoặc "cpu"
    use_reflection=True,
    detector_model_path="weights/detect_best.pt",
    vqa_model_path="weights/llava-med-mistral-v1.5-7b"
)

# Khởi tạo hệ thống
system = MedicalAISystem(config)

# Phân tích hình ảnh
result = system.analyze(
    image_path="path/to/image.jpg",
    query="Is there a polyp in this image?",
    medical_context={"patient_history": "Family history of colon cancer"}
)

# In kết quả
print(result["answer"])
print(f"Confidence: {result['answer_confidence']}")
```

### API Service

Khi chạy dưới dạng service, bạn có thể gửi requests đến API:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/image.jpg" \
  -F "query=Is there a polyp in this image?"
```

## Ưu điểm của kiến trúc LangGraph

1. **Luồng công việc rõ ràng**: Định nghĩa trực quan luồng xử lý giữa các agent
2. **Tách biệt mạch lạc**: Mỗi agent/node chỉ quan tâm đến nhiệm vụ của mình
3. **Định tuyến thông minh**: Chỉ thực thi những agent cần thiết dựa vào yêu cầu
4. **Dễ mở rộng**: Thêm node mới vào graph rất đơn giản
5. **Xử lý lỗi tốt hơn**: Lỗi được xử lý cục bộ tại từng node
6. **Khả năng checkpoint**: Lưu trạng thái của graph để khôi phục khi cần

## Phát triển

### Thêm agent mới

1. Tạo file agent mới trong thư mục `agents/`
2. Thêm node vào graph trong file `graph/pipeline.py`
3. Cập nhật router logic trong `graph/routers.py`

### Các cải tiến trong tương lai

- Thêm RAG Agent để truy xuất thông tin y khoa
- Tối ưu hóa xử lý song song giữa các node
- Tích hợp với hệ thống lưu trữ hình ảnh y tế
- Thêm cơ chế giải thích cho các polyp được phát hiện
- Hỗ trợ xử lý video nội soi

## License

[MIT License](LICENSE)
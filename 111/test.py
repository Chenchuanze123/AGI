from docx import Document

doc = Document("364号主站_Modbus_2022规约使用说明.docx")
for i, para in enumerate(doc.paragraphs[:100]):
    print(f"[{i}] {para.text}")
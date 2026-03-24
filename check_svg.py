import sys
svg_path = r'd:\___Desktop___\AutoFigure\AutoFigure-Edit\outputs\20260317_152614_07bfc6d9\template.svg'
with open(svg_path, 'r', encoding='utf-8') as f:
    content = f.read()
print(f'SVG length: {len(content)} chars')
print(f'Contains base64: {"base64" in content}')
print(f'Line count: {content.count(chr(10)) + 1}')

# 检查是否有无效字符
try:
    content.encode('utf-8')
    print('UTF-8 encoding: OK')
except:
    print('UTF-8 encoding: FAILED')

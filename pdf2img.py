# -*- coding:utf-8 -*-

# import module
from pdf2image import convert_from_path
  
  
# Almacenar el pdf con la función convert_from_path
pages = convert_from_path('test_paper.pdf')
  
for i in range(len(pages)):
   # Guardar página como imagen
   pages[i].save('page'+ str(i) +'.jpg', 'JPEG')
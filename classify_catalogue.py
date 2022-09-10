import os
import re
import xlwt


def check_file(file_path):
    os.chdir(file_path)
    print(os.path.abspath(os.curdir))
    all_file = os.listdir()
    files = []
    for f in all_file:
        if os.path.isdir(f):
            files.extend(check_file(file_path + '\\' + f))
            os.chdir(file_path)
        else:
            files.append(f)
    return files


file_list = check_file(r"C:\迅雷下载")

book = xlwt.Workbook()
sheet = book.add_sheet('文件名')
i = 0
for data in file_list:
    sheet.write(i, 0, data)
    i += 1

book.save('文件名搜索.xls')

s = ' '.join(file_list)
res_1 = re.findall(r'\D\d{8}\D', s)
print(res_1)

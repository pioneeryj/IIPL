import zipfile

def unzip_file(source_file):
    with zipfile.ZipFile(source_file, 'r') as zf:
        zipinfo = zf.infolist()

        for member in zipinfo:
            member.filename = member.filename.encode("cp437").decode("euc-kr")
            zf.extract(member)


# TL1
source_file = r'C:/Users/pione/Desktop/IIPL/#국내학회_2023/data/Synapse_raw/RawData.zip'
unzip_file(source_file)
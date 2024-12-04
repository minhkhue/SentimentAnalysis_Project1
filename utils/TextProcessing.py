import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from underthesea import word_tokenize, pos_tag, sent_tokenize
import pyvi
import regex
import string
import os
import time
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

def load_emojicon(file_path: str, mode: str = 'r', encoding: str = 'utf8') -> dict:
    """
    Nạp emojicon file vào từ điển <dict>.
    
    Tham số:
        file_path (str): Đường dẫn đến file chứa emojicon.
        mode (str): Chế độ mở file. Mặc định là 'r' (chỉ được phép đọc).
        encoding (str): Mã hóa. Mặc định là 'utf8'.
        
    Trả về:
        dict: Một bộ từ điển với key là emoji và value là ý nghĩa emoji.
    """
    try:
        with open(file_path, mode, encoding=encoding) as file:
            emoji_lst = file.read().splitlines()  # Read and split lines
            emoji_dict = {
                key: value
                for key, value in (line.split('\t') for line in emoji_lst)
            }
        return emoji_dict
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'.")
        return {}
    except Exception as e:
        print(f"Error loading emojicon file: {e}")
        return {}

def load_teencode(file_path: str, mode: str = 'r', encoding: str = 'utf8') -> dict:
    """
    Mô tả:
    ------
    Hàm này đọc file chứa các cặp từ viết tắt (teencode) và ý nghĩa tương ứng 
    (mỗi cặp được phân cách bởi dấu tab '\t'). Dữ liệu được lưu vào dictionary 
    để sử dụng trong các ứng dụng phân tích hoặc xử lý văn bản.

    Lưu ý:
    ------
    - File phải có định dạng đúng: mỗi dòng chứa một cặp key-value phân cách bằng dấu tab.
    - Bỏ qua các dòng trống hoặc không hợp lệ.
    Tham số:
    --------
    file_path : str
        Đường dẫn đến file chứa danh sách teencode.
    mode : str, tùy chọn
        Chế độ mở file, mặc định là 'r' (chế độ đọc).
    encoding : str, tùy chọn
        Kiểu mã hóa của file, mặc định là 'utf8'.

    Kết quả trả về:
    ----------------
    dict
        Một dictionary chứa các cặp key-value từ file teencode.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")
    
    teen_dict = {}
    with open(file_path, mode, encoding=encoding) as file:
        teen_lst = file.read().split('\n')
        for line in teen_lst:
            if line.strip():  # Bỏ qua các dòng trống
                key, value = line.split('\t')
                teen_dict[key] = str(value)
    return teen_dict

def load_translation(file_path: str, mode: str = 'r', encoding: str = 'utf8') -> dict:
    """
    Mô tả:
    ------
    Hàm này đọc file chứa các cặp từ tiếng Anh và tiếng Việt (mỗi cặp được phân cách bởi dấu tab '\t'). 
    Dữ liệu được lưu vào dictionary để sử dụng trong các ứng dụng xử lý ngôn ngữ tự nhiên hoặc dịch thuật.

    Lưu ý:
    ------
    - File phải có định dạng đúng: mỗi dòng chứa một cặp key-value phân cách bằng dấu tab.
    - Bỏ qua các dòng trống hoặc không hợp lệ.

    Tham số:
    --------
    file_path : str
        Đường dẫn đến file chứa danh sách từ dịch.
    mode : str, tùy chọn
        Chế độ mở file, mặc định là 'r' (chế độ đọc).
    encoding : str, tùy chọn
        Kiểu mã hóa của file, mặc định là 'utf8'.

    Kết quả trả về:
    ----------------
    dict
        Một dictionary chứa các cặp key-value từ file dịch.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")
    
    translation_dict = {}
    with open(file_path, mode, encoding=encoding) as file:
        english_lst = file.read().split('\n')
        for line in english_lst:
            if line.strip():  # Bỏ qua các dòng trống
                key, value = line.split('\t')
                translation_dict[key] = str(value)
    return translation_dict

def load_words(file_path: str, mode: str ='r', encoding: str ='utf8')-> list:
    """
    Mô tả:
    ------
    Hàm này đọc file chứa các từ, mỗi từ trên một dòng. 
    Dữ liệu được lưu vào danh sách để sử dụng trong các ứng dụng xử lý ngôn ngữ tự nhiên.
    
    Lưu ý:
    ------
    - File phải có định dạng đúng: mỗi dòng chứa một từ.
    - Bỏ qua các dòng trống hoặc không hợp lệ.
    
    Tham số:
    --------
    file_path : str
        Đường dẫn đến file chứa danh sách các từ sai.
    mode : str, tùy chọn
        Chế độ mở file, mặc định là 'r' (chế độ đọc).
    encoding : str, tùy chọn
        Kiểu mã hóa của file, mặc định là 'utf8'.

    Kết quả trả về:
    ----------------
    list
        Một danh sách chứa các từ sai chính tả.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    with open(file_path, mode, encoding=encoding) as file:
        words_list = [line.strip() for line in file if line.strip()]  # Bỏ qua các dòng trống
    return words_list

import os

def load_stopwords(file_path: str, mode: str ='r', encoding: str ='utf8')-> list:
    """
    Mô tả:
    ------
    Hàm này đọc file chứa danh sách các từ dừng tiếng Việt, mỗi từ trên một dòng.
    Dữ liệu được lưu vào danh sách để sử dụng trong các ứng dụng xử lý ngôn ngữ tự nhiên.

    Lưu ý:
    ------
    - File phải có định dạng đúng: mỗi dòng chứa một từ dừng.
    - Bỏ qua các dòng trống hoặc không hợp lệ.

    Tham số:
    --------
    file_path : str
        Đường dẫn đến file chứa danh sách từ dừng.
    mode : str, tùy chọn
        Chế độ mở file, mặc định là 'r' (chế độ đọc).
    encoding : str, tùy chọn
        Kiểu mã hóa của file, mặc định là 'utf8'.

    Kết quả trả về:
    ----------------
    list
        Một danh sách chứa các từ dừng.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    with open(file_path, mode, encoding=encoding) as file:
        stopwords_lst = [line.strip() for line in file if line.strip()]  # Bỏ qua các dòng trống
    return stopwords_lst

# Chuẩn hóa unicode tiếng việt
def loaddicchar() -> dict:
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(text: str) -> str:
    """
    Mô tả:
    ------
    - Chuyển đổi các ký tự tiếng Việt không chuẩn về dạng chuẩn.
    - Sử dụng từ điển unicode do `loaddicchar()` cung cấp.

    Tham số:
    --------
    text : str
        Văn bản cần chuẩn hóa.

    Kết quả trả về:
    ----------------
    str
        Văn bản sau khi chuẩn hóa unicode.

    """
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], text)

def process_special_word(text: str) -> str:
    """
    Mô tả:
    ------
    - Nối các từ đặc biệt phải đi kèm với một chữ sau nó thì mới biểu thị rõ thái độ.

    Tham số:
    --------
    text : str
        Văn bản cần chuẩn hóa.

    Kết quả trả về:
    ----------------
    str
        Văn bản sau khi chuẩn hóa.
    """
    # Danh sách các từ đặc biệt cần xử lý
    special_words = ['chẳng', 'không', 'chả', 'chớ', 
                     'kém', 'đáng', 'rất', 'quá', 'hơi', 
                     'thật', 'cực','khá', 'thấy', 
                     'dễ', 'hơi', 'bị', 'miễn', 'siêu', 
                     'tột', 'vô', 'hết', 'nhiều', 'ít', 
                     'khủng', 'hoàn', 'nhẹ', 'dịu', 'chưa', 'hơn', 
                     'mịn', 'châm', 'đắt', 'mắc']

    # Danh sách từ trong văn bản
    text_lst = text.split()
    new_text = []
    i = 0

    while i < len(text_lst):
        word = text_lst[i]
        if word in special_words:
            # Nếu từ hiện tại thuộc danh sách từ đặc biệt, nối với từ tiếp theo (nếu có)
            next_idx = i + 1
            if next_idx < len(text_lst):  # Đảm bảo không vượt quá danh sách
                word = f"{word}_{text_lst[next_idx]}"
                i += 1  # Bỏ qua từ tiếp theo vì đã xử lý
        # Thêm từ đã xử lý vào kết quả
        new_text.append(word)
        i += 1

    # Trả về văn bản sau khi chuẩn hóa
    return ' '.join(new_text)

def normalize_repeated_characters(text: str) -> str:
    """
    Mô tả:
    ------    
    Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    Ví dụ: "lònggggg" thành "lòng", "thiệtttt" thành "thiệt"
    
    Tham số:
    --------
        text (str): đoạn văn bản cần xử lý

    Kết quả trả về:
    ----------------
    str
        Đoạn văn bản đã được xử lý
    """
    return regex.sub(r'(.)\1+', r'\1', text)

def process_postag_thesea(text: str):
    """
    Mô tả:
    ------    
    Tìm kiếm ra các từ ghép đặc biệt
    
    Tham số:
    --------
        text (str): đoạn văn bản cần xử lý

    Kết quả trả về:
    ----------------
    str
        Đoạn văn bản đã được xử lý
    """
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        word_token = str(word_tokenize(sentence, format='text'))
        new_text = process_special_word(word_token)
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(new_text))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

def remove_stopword(text: str, stopwords: list):
    """
    Mô tả:
    ------    
    Tìm kiếm ra các stopword và loại bỏ chúng
    
    Tham số:
    --------
        text (str): đoạn văn bản cần xử lý

    Kết quả trả về:
    ----------------
    str
        Đoạn văn bản đã được xử lý
    """
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

# Xóa HTML code
def remove_html(text) -> str:
    return regex.sub(r'<[^>]*>', '', text)

# Tìm và đếm các từ trong list_of_words
def find_words(text: str, list_of_words: list) -> tuple:
    """
    Mô tả:
    ------    
    Tìm và đếm các từ trong list_of_words
    
    Tham số:
    --------
        text (str): đoạn văn bản cần xử lý
        list_of_words (list): list các từ cần tìm kiếm
    Kết quả trả về:
    ----------------
    str
        Tuple chứa số lượng từ, và list các từ đó
    """
    text_lower = text.lower()
    word_count = 0
    word_list = []
    for word in list_of_words:
        if word in text_lower:
            word_list.append(word)
    word_count = len(word_list)
    return word_count, word_list

def process_text(text: str, emoji_dict: dict, teen_dict: dict, translate_dict: dict, wrong_lst: list, stopwords_lst: list,) -> str:
    """
    Mô tả:
    ------    
    Hàm này sử dụng để xử lý ngôn ngữ dựa trên câu từ được cung cấp (text), emoji, teencode, wrong-word, stop-words, loại bỏ punctuations, khoảng trắng thừa
    
    Tham số:
    --------
        text (str): đoạn văn bản cần xử lý
        emoji_dict (dict): từ điển emoji
        teen_dict (dict): từ điển teencode
        translate_dict (dict): từ điển chuyển đổi Anh-Việt
        wrong_lst (list): list các wrong-word
        stopwords_lst (list): list các từ vô nghĩa

    Kết quả trả về:
    ----------------
    str
        Đoạn văn bản đã được xử lý loại bỏ punctuations, khoảng trắng thừa
    """
    document = text.lower()
    document = document.replace("’",'') # thay thế dấu nháy đơn (')= ký tự rỗng =>xóa dấu nháy đơn
    # document = regex.sub(r'\.+', ".", document) #thay thế các chuỗi có 1 hay nhiều dấu (.) = 1 dấu (.)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        ###### Chuyển đổi các từ tiếng Anh -> Việt (nếu có)
        sentence = ' '.join(translate_dict[word] if word in translate_dict else word for word in sentence.split())
        # ###### DEL wrong words
        # sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        ###### Xử lý chữ đuôi kéo dài như chánnnn,...
        sentence = normalize_repeated_characters(sentence)
        ###### Xử lý từ ghép để có nghĩa
        sentence = process_postag_thesea(sentence)
        ###### Xóa stop-word
        sentence = ' '.join('' if word in stopwords_lst else word for word in sentence.split())
        new_sentence = new_sentence + sentence + '. '
    document = new_sentence
    #print(document)
    ###### Loại bỏ khoảng trắng thừa trong câu và punctuations
    document = regex.sub(r'[^\w\s]', '', document)  # Loại bỏ punctuations
    document = regex.sub(r'\s+', ' ', document).strip()  # Loại bỏ khoảng trắng thừa
    return document

# Phân tích 1 biến số

# Phân tích 1 biến số 
def phan_tich_1_bien_so(data: pd.DataFrame, column_name, lower_quantile: float=0.25, upper_quantile:float=0.75, threshold:float=1.5, plot_fig:bool=True, fig_length:float=5.4, fig_width:float=5.4, hue=None, sharex:bool=True, sharey:bool=False, bins='auto'): 
    """
    Mô tả:
    ------    
    Hàm này sử dụng để phân tích biến số liệu bằng phương pháp tứ phân vị và boxplot, kèm trực quan dữ liệu
    
    Args:
        data (pd.DataFrame): Bảng dữ liệu pandas DataFrame
        column_name (str): Tên của cột biến số liệu cần xử lý
        lower_quantile (float, optional): Giá trị tứ phân vị ở Q1. Defaults to 0.25.
        upper_quantile (float, optional): Giá trị tứ phân vị ở Q3. Defaults to 0.75.
        threshold (float, optional): Hệ số ngưỡng. Defaults to 1.5.
        plot_fig (bool, optional): Trực quan dữ liệu. Defaults to True.
        fig_length (float, optional): Kích thước dài của biểu đồ. Defaults to 5.4.
        fig_width (float, optional): Kích thước cao của biểu đồ. Defaults to 5.4.
        hue (_type_, optional): Phân loại biến dữ liệu. Defaults to None.
        sharex (bool, optional): Các biểu đồ trực quan chia sẻ cùng kích thước trục x. Defaults to True.
        sharey (bool, optional): Các biểu đồ trực quan chia sẻ cùng kích thước trục y. Defaults to False.
        bins (str, optional): Binning dữ liệu. Defaults to 'auto'.
    """
    df = data.copy()
    # Tính toán các giá trị thống kê
    skew = df[column_name].skew() 
    kurtosis = df[column_name].kurtosis() 
    var = df[column_name].var()
    std = df[column_name].std()
    minnimum = df[column_name].min()
    mean = df[column_name].mean()
    median = df[column_name].median()
    mode = df[column_name].mode()[0]
    max = df[column_name].max()
    Q1 = df[column_name].quantile(lower_quantile)
    Q3 = df[column_name].quantile(upper_quantile)
    IQR = Q3 - Q1
    upper_bound = Q3 + threshold * IQR
    lower_bound = Q1 - threshold * IQR
    if lower_bound < minnimum:
        lower_bound = minnimum
    upper_count = df[df[column_name] > upper_bound].shape[0]
    lower_count = df[df[column_name] < lower_bound].shape[0]
    total_outliers = upper_count + lower_count
    outliers_percentage = (total_outliers/df.shape[0])*100

    # Tạo bảng dữ liệu thống kê
    stats_df = pd.DataFrame({
        'Thống kê': ['Skew', 'Kurtosis', 'Phương sai', 'Độ lệch chuẩn', 'Min', 'Mean', 'Mode', 'Max', 
                      f'Q1 ({int(lower_quantile*100)}%)', 'Median (Q2)', f'Q3 ({int(upper_quantile*100)}%)', 'IQR', 
                      'Cận râu trên', 'Cận râu dưới', 'Outliers trên', 'Outliers dưới', 'Outlier (%)'],
        'Giá trị': [f"{skew:.2f}", f"{kurtosis:.2f}", f"{var:.2f}", f"{std:.2f}", f"{minnimum:.2f}", f"{mean:.2f}", f"{mode:.2f}", f"{max:.2f}",
                    f"{Q1:.2f}", f"{median:.2f}", f"{Q3:.2f}", f"{IQR:.2f}", f"{upper_bound:.2f}", f"{lower_bound:.2f}",
                    f"{upper_count:.2f}", f"{lower_count:.2f}", f"{outliers_percentage:.2f}"]
    })
        # Nhận xét về các giá trị thống kê
    print(f'Nhận xét dữ liệu {column_name}'.center(70,'-'))
    if skew > 0:
        print(f'''Skew = {skew:.2f} > 0 => Phân phối lệch phải.''')
    elif skew == 0:
        print('''Skew = 0 => Phân phối chuẩn.''')
    else:
        print(f'''Skew = {skew:.2f} < 0 => Phân phối lệch trái.''')

    if kurtosis > 0:
        print(f'''Kurtosis = {kurtosis:.2f} > 0 => Phân phối nhọn hơn phân phối chuẩn.''')
    elif kurtosis == 0:
        print('''Kurtosis = 0 => Phân phối có độ nhọn như phân phối chuẩn.''')
    else:
        print(f'''Kurtosis = {kurtosis:.2f} < 0 => Phân phối không nhọn bằng phân phối chuẩn.''')
    print(f'''Phương sai: {var:.2f} => {'Dữ liệu có sự phân tán lớn' if var > mean else 'Dữ liệu có sự phân tán nhỏ'}.''')
    print(f'''Độ lệch chuẩn: {std:.2f} => {'Dữ liệu có sự biến động lớn' if std > mean else 'Dữ liệu có sự biến động nhỏ'}.''')
    print(f'Bảng thống kê của {column_name}'.center(70,'-'))
    print(stats_df.to_string(index=True))
    print(f'Kết thúc phân tích {column_name}'.center(70,'-'))
        # Vẽ biểu đồ
    if plot_fig == True:
        print(f'Vẽ biểu đồ {column_name}'.center(70,'-'))
        fig, axes = plt.subplots(2, 1, figsize=(fig_length, fig_width), sharex=sharex, sharey=sharey)
        sns.boxplot(data=df, x=column_name, hue=hue, ax=axes[0])
        sns.histplot(data=df, x=column_name, hue=hue, kde=True, ax=axes[1], bins=bins)
        if hue == None:
            plt.axvline(x=median, ymin = 0, color = 'r')
            plt.axvline(x=upper_bound, ymin = 0, color = 'b')
            if lower_bound < minnimum:
                plt.axvline(x=minnimum, ymin = 0, color = 'b')
            else:
                plt.axvline(x=lower_bound, ymin = 0, color = 'b')
        plt.show()
    return

def create_sentiment_col(target, stars):
    """
    Mô tả:
    ------    
    Hàm này sử dụng để binning số sao thành Positive/Negative sentinment

    Args:
        target: Nội dung của biến dữ liệu cần binning
        stars: Số sao dùng để đánh giá phân loại
        VD: stars = 3, những giá trị 'so_sao' từ 3 trở xuống là Negative, còn lại là Positive

    Returns:
        Giá trị sentiment ở dạng Negative/Positive
    """
    if int(target) <= stars:
        return 'Negative'
    else:
        return 'Positive'
    
    import time
import regex
from underthesea import word_tokenize, chunk

def txt_process_for_cols(input_df, 
                         input_col_name:str, 
                         emoji_dict: dict, 
                         teen_dict:dict, 
                         translate_dict:dict, 
                         stopwords_lst:list, 
                         data_col:list = [None], 
                         groupby:bool = True, 
                         eng_vie: bool = True, 
                         chunking:bool = True) -> pd.DataFrame:
    """
    Mô tả:
    ------   
    Hàm xử lý nội dung câu chữ, tokenize một cột dữ liệu của Dataframe.

    Parameters:
    - input_df: pd.DataFrame, dataframe chứa nội dung cần xử lý.
    - input_col_name: str, cột dữ liệu cần được NLP
    - data_col: list, dữ liệu list tên các cột để sử dụng cho groupby
    - emoji_dict: dict, từ điển emoji để chuyển đổi.
    - teen_dict: dict, từ điển teencode để chuyển đổi.
    - translate_dict: dict, từ điển dịch từ tiếng Anh sang tiếng Việt.
    - stopwords_lst: list, danh sách stopwords cần loại bỏ.
    - groupby: bool, Mặc định = True, sẽ groupby dữ liệu theo các biến trong 'data_col' và gộp dữ liệu text ở cột 'input_col_name'
    - eng_vie: bool = True, , 
    - chunking: bool, Mặc định = True, sử dụng để chunk đoạn text thành các từ/từ ghép kèm pos_tag, phục vụ phân tích và trích xuất các từ/từ ghép có nghĩa cần thiết

    Returns:
    - dataframe: pd.DataFrame, dataframe với input_col_name đã được xử lý nội dung chữ từng bước, mỗi bước ra một cột.
    """
    start_process = time.time()
    df = input_df.copy()
    i = 1
    if groupby:
        # Step 1: Gộp nội dung bình luận theo 'input_col_name'
        start = time.time()
        processed_col = 'processed_'+input_col_name
        df = df.groupby(data_col)[input_col_name]\
            .apply(' '.join).reset_index()
        df.rename(columns={input_col_name: processed_col}, inplace=True)
        print(f'Step {i}: Gộp nội dung cột "{input_col_name}"           --- xử lý trong {(time.time() - start):.2f}s')
        i += 1
    else:
        processed_col = input_col_name
        pass

    # Step 2: Xử lý lowercase và convert unicode
    start = time.time()
    df[processed_col+'_lowercase_convert_unicode'] = df[processed_col]\
        .apply(lambda txt: covert_unicode(txt.lower()))
    ## Bỏ một số câu từ lặp di lặp lại, không cần thiết
    unusable_string1 = "’"
    unusable_string2 = 'Làm sao để phân biệt hàng có trộn hay không ?\nHàng trộn sẽ không thể xuất hoá đơn đỏ (VAT) 100% được do có hàng không nguồn gốc trong đó.\nTại Hasaki, 100% hàng bán ra sẽ được xuất hoá đơn đỏ cho dù khách hàng có lấy hay không. Nếu có nhu cầu lấy hoá đơn đỏ, quý khách vui lòng lấy trước 22h cùng ngày. Vì sau 22h, hệ thống Hasaki sẽ tự động xuất hết hoá đơn cho những hàng hoá mà khách hàng không đăng kí lấy hoá đơn.\nDo xuất được hoá đơn đỏ 100% nên đảm bảo 100% hàng tại Hasaki là hàng chính hãng có nguồn gốc rõ ràng.'
    unusable_string3 = 'THÔNG TIN SẢN PHẨM'
    unusable_string4 = '1. '
    unusable_string5 = '*Lưu ý: Tác dụng có thể khác nhau tuỳ cơ địa của người dùng'
    unusable_string6 = '\n'
    df[processed_col+'_lowercase_convert_unicode'] = df[processed_col+'_lowercase_convert_unicode']\
        .apply(lambda txt: txt.lower()
           .replace(unusable_string1.lower(), '')
           .replace(unusable_string2.lower(), '')
           .replace(unusable_string3.lower(), '')
           .replace(unusable_string4.lower(), '')
           .replace(unusable_string5.lower(), '')
           .replace(unusable_string6.lower(), ' '))
    print(f'Step {i}: Xử lý lowercase, convert unicode, xóa từ không cần thiết  --- xử lý trong {(time.time() - start):.2f}s')
    i += 1

    # Step 3: Xử lý emoji thành chữ
    start = time.time()
    df[processed_col+'_emoji_convert'] = df[processed_col+'_lowercase_convert_unicode']\
        .apply(lambda txt: ''.join(emoji_dict[word] + ' ' if word in emoji_dict else word for word in list(txt)))
    print(f'Step {i}: Chuyển đổi emoji thành chữ                      --- xử lý trong {(time.time() - start):.2f}s')
    i += 1

    # Step 4: Xử lý teencode
    start = time.time()
    df[processed_col+'_teencode_convert'] = df[processed_col+'_emoji_convert']\
        .apply(lambda txt: ' '.join(teen_dict[word] if word in teen_dict else word for word in txt.split()))
    print(f'Step {i}: Chuyển đổi teencode thành chữ                   --- xử lý trong {(time.time() - start):.2f}s')
    i += 1

    # Step 5: Xóa dấu câu và khoảng trắng thừa
    pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
    start = time.time()
    df[processed_col+'_puntuation_space_remove'] = df[processed_col+'_teencode_convert']\
        .apply(lambda txt: ' '.join(regex.findall(pattern, txt)))
    print(f'Step {i}: Xóa dấu câu và khoảng trắng thừa                --- xử lý trong {(time.time() - start):.2f}s')
    i += 1

    if groupby:
    # Step 6: Dịch từ tiếng Anh sang tiếng Việt
        start = time.time()
        df[processed_col+'_eng_vie'] = df[processed_col+'_puntuation_space_remove']\
            .apply(lambda txt: ' '.join(translate_dict[word] if word in translate_dict else word for word in txt.split()))
        print(f'Step {i}: Dịch từ tiếng Anh sang tiếng Việt               --- xử lý trong {(time.time() - start):.2f}s')
        i += 1
    else:
        pass
    next_input_name_step = processed_col+'_puntuation_space_remove'

    # Step 7: Xử lý chữ đuôi kéo dài
    start = time.time()
    df[processed_col+'_remove_word_'] = df[next_input_name_step]\
        .apply(lambda txt: normalize_repeated_characters(txt))
    print(f'Step {i}: Chữ đuôi kéo dài                                --- xử lý trong {(time.time() - start):.2f}s')
    i += 1

    # Step 8: Word tokenize
    start = time.time()
    df[processed_col+'_special_words_token'] = df[processed_col+'_remove_word_']\
        .apply(lambda txt: word_tokenize(sentence=txt, format='text'))
    print(f'Step {i}: Word tokenize                                   --- xử lý trong {(time.time() - start):.2f}s')
    i += 1

    # Step 9: Xử lý stopwords
    start = time.time()
    df[processed_col+'_special_words_remove_stopword'] = df[processed_col+'_special_words_token']\
        .apply(lambda txt: ' '.join('' if word in stopwords_lst else word for word in txt.split()))
    ## Xóa dấu câu
    df[processed_col+'_special_words_remove_stopword'] = df[processed_col+'_special_words_remove_stopword']\
        .apply(lambda txt: regex.sub(r'[^\w\s]', '', txt).strip())
    ## Xóa khoảng trống thừa
    df[processed_col+'_special_words_remove_stopword'] = df[processed_col+'_special_words_remove_stopword']\
        .apply(lambda txt: regex.sub(r'\s+', ' ', txt).strip())
    print(f'Step {i}: Xóa stopwords, dấu câu còn lại, khoảng trắng thừa --- xử lý trong {(time.time() - start):.2f}s')
    i += 1

    if chunking:
        # Step 10: Chunking
        start = time.time()
        df[processed_col+'_chunk'] = df[processed_col+'_special_words_remove_stopword']\
            .apply(lambda txt: chunk(sentence=txt, format='text'))
        print(f'Step {i} Chunking ra các từ/từ ghép kèm pos_tag của nó   --- xử lý trong {(time.time() - start):.2f}s')
        print(f'---------------------------------')
        print(f'Toàn bộ xử lý trong {(time.time() - start_process):.2f}s')
    else:
        pass
    return df

def sentiment_predict(input_df: pd.DataFrame, text_col_name: str, trained_tfidf):
    tfidf_matrix = trained_tfidf.transform(input_df[text_col_name])
    """
    Mô tả:
    --------
    Sử dụng mô hình tfidf đã trên trên bộ dữ liệu để xử lý ra bộ vector chữ
    
    Parameters
    --------
    - input_df: Bảng dataframe chứa cột dữ liệu cần xử lý vectorizer
    - text_col_name: Tên của cột dữ liệu cần xử lý
    
    Returns:
    - tập dữ liệu vector chữ 'vec_assembler'
    """
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=trained_tfidf.get_feature_names_out())

    scaler = MinMaxScaler()
    input_df['txt_length'] = input_df[text_col_name].apply(len)
    input_df['scaled_txt_length'] = scaler.fit_transform(input_df[['txt_length']])

    vec_assembler = pd.concat([tfidf_df, input_df['scaled_txt_length']], axis=1)
    return vec_assembler
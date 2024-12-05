import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from underthesea import word_tokenize, pos_tag, sent_tokenize
import pyvi
from utils import TextProcessing as tpr
import regex
import string
import os
import time
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

import surprise
from surprise.model_selection import cross_validate
from surprise.model_selection.validation import cross_validate
from surprise import Reader, Dataset, SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore, CoClustering, BaselineOnly

from utils import TextProcessing as tpr
from utils import evaluation

def compare_surprise_models(input_df, userId_col_name, productId_col_name, product_name_col_name, rating_col_name):
    """
    So sánh các mô hình của Surprise dựa trên RMSE, MAE và thời gian cross-validate.

    Parameters:
    -----------
    input_df : pd.DataFrame
        DataFrame chứa dữ liệu đầu vào.
    userId_col_name : str
        Tên cột chứa mã người dùng.
    productId_col_name : str
        Tên cột chứa mã sản phẩm.
    product_name_col_name : str
        Tên cột chứa tên sản phẩm.
    rating_col_name : str
        Tên cột chứa đánh giá của người dùng.

    Returns:
    --------
    pd.DataFrame
        DataFrame chứa kết quả so sánh các mô hình.

    Ví dụ:
    --------    
    # Giả sử bạn có dữ liệu đầu vào
    result_df = compare_surprise_models(
        input_df=df, 
        userId_col_name='ma_khach_hang', 
        productId_col_name='ma_san_pham', 
        product_name_col_name='ten_san_pham', 
        rating_col_name='so_sao'
)
    """
    # Chuẩn bị dữ liệu
    df = input_df.copy()
    df.rename(columns={userId_col_name: 'userId', 
                       productId_col_name: 'productId', 
                       product_name_col_name: 'product_name',
                       rating_col_name: 'rating'}, inplace=True)
    df = df[['userId', 'productId', 'product_name', 'rating']].reset_index(drop=True)
    df['userId'] = df.userId.astype('int')
    df['productId'] = df.productId.astype('int')
    df['rating'] = df.rating.astype('float')

    # Danh sách thuật toán
    algorithms = {
        "SVD": SVD(),
        "SVDpp": SVDpp(),
        "NMF": NMF(),
        "SlopeOne": SlopeOne(),
        "KNNBasic": KNNBasic(),
        "KNNBaseline": KNNBaseline(),
        "KNNWithMeans": KNNWithMeans(),
        "KNNWithZScore": KNNWithZScore(),
        "CoClustering": CoClustering(),
        "BaselineOnly": BaselineOnly(),
    }

    # Chuẩn bị dữ liệu cho Surprise
    reader = Reader()
    data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)

    # So sánh các mô hình
    results = []
    for name, algorithm in algorithms.items():
        start_time = time.time()
        scores = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        elapsed_time = time.time() - start_time
        results.append({
            "Model": name,
            "RMSE": scores['test_rmse'].mean(),
            "MAE": scores['test_mae'].mean(),
            "Cross-Validation Time (s)": elapsed_time
        })

    # Trả về DataFrame kết quả
    return pd.DataFrame(results).sort_values(by=['RMSE', 'Cross-Validation Time (s)'], ascending=True).reset_index(drop=True)

def surprise_model_builder(input_df=pd.DataFrame(), 
                           userId_col_name: str = '', 
                           productId_col_name: str = '', 
                           product_name_col_name: str = '', 
                           rating_col_name: str = '', 
                           model_algorithm: str = ''):
    # Sao chép dữ liệu
    start_process = time.time()
    df = input_df.copy()
    
    # Đổi tên các cột
    df.rename(columns={userId_col_name: 'userId', 
                       productId_col_name: 'productId', 
                       product_name_col_name: 'product_name',
                       rating_col_name: 'rating', }, inplace=True)
    
    # Lựa chọn các cột để đưa vào surprise xử lý
    df = df[['userId', 'productId', 'product_name', 'rating']]

    # Chuyển đổi loại dữ liệu
    df['userId'] = df.userId.astype('int')
    df['productId'] = df.productId.astype('int')
    df['rating'] = df.rating.astype('float')

    # Từ điển ánh xạ thuật toán
    algorithms = {
        "SVD": SVD(),
        "SVDpp": SVDpp(),
        "NMF": NMF(),
        "SlopeOne": SlopeOne(),
        "KNNBasic": KNNBasic(),
        "KNNBaseline": KNNBaseline(),
        "KNNWithMeans": KNNWithMeans(),
        "KNNWithZScore": KNNWithZScore(),
        "CoClustering": CoClustering(),
        "BaselineOnly": BaselineOnly(),
    }

    # Kiểm tra model_algorithm
    if model_algorithm not in algorithms:
        raise ValueError(f"Thuật toán '{model_algorithm}' không được hỗ trợ. Hãy chọn một trong các thuật toán sau: {list(algorithms.keys())}")

    # Chọn thuật toán
    algorithm_ = algorithms[model_algorithm]

    # Kiểm tra và đánh giá thuật toán với cross-validation
    reader = Reader()
    data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)
    results = cross_validate(algorithm_, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)

    # Kiểm tra độ phù hợp của thuật toán
    rmse_mean = results['test_rmse'].mean()
    if rmse_mean <= 1.0:
        pass
    else:
        raise ValueError(f"\nThuật toán '{model_algorithm}' không phù hợp vì RMSE trung bình = {rmse_mean:.2f} > 1. Hãy chọn thuật toán khác.\n")

    # Huấn luyện mô hình trên toàn bộ dữ liệu
    trainset = data.build_full_trainset()
    algorithm_.fit(trainset)
    evaluation.Save_Object(algorithm_,f'{model_algorithm}_algo.pkl')
    print(f'\nToàn bộ funtion xử lý trong {(time.time() - start_process):.2f}s')
    print(f'\nTrung bình RMSE của thuật toán {model_algorithm} = {rmse_mean:.2f}\n')
    return algorithm_

def surprise_recommendation(input_df=pd.DataFrame(), userId_col_name: str = '', productId_col_name: str = '', product_name_col_name: str = '', 
                 rating_col_name: str = '', model_algorithm=None, userId: int = 0, 
                 rate_threshold: float = 4, top_recommend: int = 5, user_history:bool = False):
    """
    Mô tả:
    ------
    Hàm recommend sản phẩm cho người dùng dựa trên Surprise Library.

    Params:
        - input_df (DataFrame): Dữ liệu đầu vào chứa thông tin userId, productId và rating.
        - userId_col_name (str): Tên cột chứa mã người dùng ở dataframe đầu vào.
        - productId_col_name (str): Tên cột chứa mã sản phẩm ở dataframe đầu vào
        - rating_col_name (str): Tên cột chứa điểm đánh giá ở dataframe đầu vào
        - model_algorithm (str): Tên của biến model được train từ function 'surprise_model_builder'
        - userId (int): ID của người dùng muốn recommend.
        - rate_threshold (float): Ngưỡng điểm đánh giá để lọc kết quả (default = 4).
        - top_recommend (int): Số lượng sản phẩm recommend tối đa(default = 5).
        - user_history (bool): Lịch sử mua hàng của khách hàng userId này.

    Returns:
        - Bảng kết quả gợi ý sản phẩm theo userId.
    """
    # Sao chép dữ liệu
    start_process = time.time()
    df = input_df.copy()
    
    # Đổi tên các cột
    df.rename(columns={userId_col_name: 'userId', 
                       productId_col_name: 'productId', 
                       product_name_col_name: 'product_name',
                       rating_col_name: 'rating', }, inplace=True)
    
    # Lựa chọn các cột để đưa vào surprise xử lý
    df = df[['userId', 'productId', 'product_name', 'rating']]

    # Chuyển đổi loại dữ liệu
    df['userId'] = df.userId.astype('int')
    df['productId'] = df.productId.astype('int')
    df['rating'] = df.rating.astype('float')

    # Chọn thuật toán
    algorithm_ = model_algorithm

    # Dự đoán điểm đánh giá cho sản phẩm
    df_score = df[['productId','product_name']].drop_duplicates()
    df_score['EstimateScore'] = df_score['productId'].apply(lambda x: algorithm_.predict(userId, x).est) # type: ignore
    df_score.sort_values(by=['EstimateScore'], ascending=False, inplace=True)

    # Lịch sử mua hàng của userId
    if user_history == True:
        df_select = df[(df['userId'] == userId) & (df['rating'] >= rate_threshold)]
        df_select = df_select.set_index('productId')
        #df_select = df_select.join(df_title)['Name']
        print(f'\n\nLịch sử mua hàng của userId = {userId}.')
        display(df_select[['product_name', 'rating']].head(df_select.shape[0])) # type: ignore
    
    # Lọc kết quả theo ngưỡng đánh giá
    recommend = df_score[df_score['EstimateScore'] >= rate_threshold].head(top_recommend).set_index('productId')
    print(f'\n\nCác sản phẩm recommend cho userId = {userId}.')
    print(f'\nToàn bộ funtion xử lý trong {(time.time() - start_process):.2f}s')
    return recommend

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from utils import TextProcessing as tpr
from utils import evaluation
from underthesea import word_tokenize, pos_tag, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
from wordcloud import WordCloud as wc
label_encoder = LabelEncoder()

# Load các emoji biểu cảm thường gặp
emoji_dict = tpr.load_emojicon(file_path='files/emojicon.txt')
teen_dict = tpr.load_teencode(file_path='files/teencode.txt')
translate_dict = tpr.load_translation(file_path='files/english-vnmese.txt')
stopwords_lst = tpr.load_stopwords(file_path='files/vietnamese-stopwords.txt')
wrong_lst = tpr.load_words(file_path='files/wrong-word.txt')
# Nạp các từ ngữ tiêu cực sau khi đã xử lý bằng tay
positive_words_lst: list = tpr.load_words(file_path='files/hasaki_positive_words.txt')
# Nạp các từ ngữ tiêu cực sau khi đã xử lý bằng tay
negative_words_lst: list = tpr.load_words(file_path='files/hasaki_negative_words.txt')

# Load model và TF-IDF vectorizer
@st.cache_resource
def load_model_and_tfidf():
    model = evaluation.Load_Object('models/proj1_sentiment_lgr_model.pkl')
    tfidf_vectorizer = evaluation.Load_Object('models/proj1_tfidf_vectorizer.pkl')
    return model, tfidf_vectorizer

# Load model và tfidf
proj1_sentiment_lgr_model, proj1_tfidf_vectorizer = load_model_and_tfidf()

# Giao diện phần 'Tải dữ liệu lên hệ thống'
st.sidebar.title('Menu:')
info_options = st.sidebar.radio(
    'Các chức năng :robot_face:', 
    options=['Tổng quan về hệ thống', 'Tải dữ liệu lên hệ thống', 'Tổng quan về dataset', 'Thông tin về sản phẩm', 'Dự báo thái độ cho dataset', 'Dự báo thái độ cho comment']
)
st.sidebar.write('-'*3)
st.sidebar.write('#### :star: Giảng viên hướng dẫn:')
st.sidebar.write(':female-teacher: Thạc Sỹ Khuất Thùy Phương')
st.sidebar.write('-'*3)
st.sidebar.write('#### Nhóm cùng thực hiện:')
st.sidebar.write(' :boy: Nguyễn Minh Trí')
st.sidebar.write(' :boy: Võ Huy Quốc')
st.sidebar.write(' :boy: Phan Trần Minh Khuê')
st.sidebar.write('-'*3)
st.sidebar.write('#### :clock830: Thời gian thực hiện:')
st.sidebar.write(':spiral_calendar_pad: 14/12/2024')

## Kiểm tra dữ liệu đã upload trước đó
if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None  # Khởi tạo nếu chưa có dữ liệu
    
## Các bước thực hiện
if info_options == 'Tổng quan về hệ thống':
    st.image('img/hasaki_logo.png', use_column_width=True)
    general_info_tabs = st.tabs(['Business Objective', 'Triển khai hệ thống'])
    with general_info_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.image('img/think_man.png', use_column_width=True)
        with col2:
            st.write('- HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống cửa hàng trải dài trên toàn quốc; và hiện đang là đối tác phân phối chiến lược tại thị trường Việt Nam của hàng loạt thương hiệu lớn...”')
            st.write('- Khách hàng có thể vào website hasaki.vn để tìm kiếm, lựa chọn sản phẩm, xem các đánh giá/nhận xét và đặt mua sản phẩm.')
            st.write('- Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhãn hàng hiểu khách hàng rõ hơn, biết họ đánh giá gì về sản phẩm, từ đó có thể cải thiện chất lượng sản phẩm cũng như các dịch vụ đi kèm.')
    with general_info_tabs[1]:
        st.header('Quy trình triển khai hệ thống')
        st.image('img/Gioi_thieu_proj1.PNG', use_column_width=True)

## Xem dữ liệu đã upload lên, đưa dữ liệu vào session để sử dụng lại được
if info_options == 'Tải dữ liệu lên hệ thống':
    st.image('img/hasaki_logo.png', use_column_width=True)
    st.header('Tải dữ liệu đầu vào')

    # Chỉ hiện nút tải file nếu chưa có dữ liệu
    if st.session_state['uploaded_data'] is None:
        uploaded_file = st.file_uploader('Upload file CSV chứa dữ liệu:', type='csv')

        if uploaded_file is not None:
            # Đọc file CSV và lưu vào session_state
            data = pd.read_csv(uploaded_file)
            data = data.drop(data[data['ngay_binh_luan'] == '30/11/-0001'].index)
            data['ngay_binh_luan'] = pd.to_datetime(data['ngay_binh_luan'], format='%Y-%m-%d')
            data['quarter'] = data['ngay_binh_luan'].dt.to_period('Q').astype(str)
            st.session_state['uploaded_data'] = data
            st.write('-'*3)
            st.success('Dữ liệu đã được tải lên!')
            st.dataframe(data[['ma_khach_hang', 'ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'so_sao', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gia_ban']].head(5))
            st.dataframe(data[['ma_khach_hang', 'ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'so_sao', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gia_ban']].tail(5))  
    else:
        st.info('Dữ liệu đã được tải lên trước đó.')
        data = st.session_state['uploaded_data']  # Lấy dữ liệu từ session_state
        st.dataframe(data[['ma_khach_hang', 'ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'so_sao', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gia_ban']].head(5))
        st.dataframe(data[['ma_khach_hang', 'ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'so_sao', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gia_ban']].tail(5))
# Giao diện phần 'Tổng quan về dataset'
if info_options == 'Tổng quan về dataset':
    st.image('img/hasaki_logo.png', use_column_width=True)
    if st.session_state['uploaded_data'] is None:
        st.warning('Dataset chưa được tải lên')
    else:
        data = st.session_state['uploaded_data']  # Lấy dữ liệu từ session_state
        # Tạo đồ thị đếm số sao
        st.write("### Phân bổ số sao đánh giá")
        fig, ax = plt.subplots(figsize=(10, 6))  # Khởi tạo figure và axis cho Seaborn
        sns.countplot(data=data, x='so_sao', hue='so_sao', palette='tab10', ax=ax)  # Tạo biểu đồ trên ax
        # Thêm số count trên cột
        for container in ax.containers: # type: ignore
            ax.bar_label(container)
        # Tùy chỉnh đồ thị
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Xoay nhãn trục X
        ax.set_title('Số sao phân bổ')  # Tiêu đề
        plt.tight_layout()  # Đảm bảo bố cục không bị cắt
        # Hiển thị đồ thị trên Streamlit
        st.pyplot(fig)
        
        # Số lượng các từ tích cực
        st.write("### Số lượng các từ tích cực đã nhận xét")
        # Tạo đồ thị
        fig, ax = plt.subplots(figsize=(10, 6))  # Khởi tạo figure và axis
        sns.countplot(data=data, x='positive_words_count', hue='positive_words_count', legend=False, palette='tab10', ax=ax)  # type: ignore # Tạo biểu đồ
        # Thêm nhãn trên các cột
        for container in ax.containers: # type: ignore
            ax.bar_label(container)
        # Tùy chỉnh đồ thị
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Xoay nhãn trục X
        ax.set_title('Số lượng các từ tích cực đã nhận xét')  # Thêm tiêu đề
        plt.tight_layout()  # Đảm bảo bố cục gọn gàng
        # Hiển thị đồ thị trên Streamlit
        st.pyplot(fig)

        # Số lượng các từ tiêu cực
        st.write("### Số lượng các từ tiêu cực đã nhận xét")
        # Tạo đồ thị
        fig, ax = plt.subplots(figsize=(10, 6))  # Khởi tạo figure và axis
        sns.countplot(data=data, x='negative_words_count', hue='negative_words_count', legend=False, palette='tab10', ax=ax)  # type: ignore # Tạo biểu đồ
        # Thêm nhãn trên các cột
        for container in ax.containers: # type: ignore
            ax.bar_label(container)
        # Tùy chỉnh đồ thị
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Xoay nhãn trục X
        ax.set_title('Số lượng các từ tiêu cực đã nhận xét')  # Thêm tiêu đề
        plt.tight_layout()  # Đảm bảo bố cục gọn gàng
        # Hiển thị đồ thị trên Streamlit
        st.pyplot(fig)

        # Thống kê số lượng bình luận theo quý
        # Tạo cột 'quarter' (quý) từ cột 'ngay_binh_luan'
        # Chuyển dạng dữ liệu biến 'ngay_binh_luan' sang biến datetime
        # Dữ liệu thời gian bất thường.
        # data = data.drop(data[data['ngay_binh_luan'] == '30/11/-0001'].index)
        # data['ngay_binh_luan'] = pd.to_datetime(data['ngay_binh_luan'], format='%Y-%m-%d')
        # data['quarter'] = data['ngay_binh_luan'].dt.to_period('Q').astype(str)
        comment_count_quarter = data.groupby('quarter').size().reset_index()
        comment_count_quarter.rename(columns={0: 'so_luong_binh_luan_quy'}, inplace=True)
        st.write("### Thống kê số lượng bình luận theo quý")
        # Tạo figure và axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Tạo biểu đồ lineplot
        sns.lineplot(
            data=comment_count_quarter, 
            x='quarter', 
            y='so_luong_binh_luan_quy', 
            marker='o', 
            label='Số lượng bình luận theo quý', 
            ax=ax
        )
        # Thêm giá trị trực tiếp lên biểu đồ
        for x, y in zip(comment_count_quarter['quarter'], comment_count_quarter['so_luong_binh_luan_quy']):
            ax.text(x, y, str(y), color='black', ha='center', va='bottom', fontsize=10)
        # Thiết lập tiêu đề và nhãn
        ax.set_title('Thống kê số lượng bình luận theo quý', fontsize=16)
        ax.set_xlabel('Quý', fontsize=12)
        ax.set_ylabel('Số lượng bình luận theo quý', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.grid(True)
        ax.legend()
        # Hiển thị biểu đồ
        st.pyplot(fig)

        # Tần suất Positive/Negative
        st.write("### Tần suất Positive/Negative trên tập dữ liệu")
        # Tạo figure và axis
        fig, ax = plt.subplots(figsize=(8, 5))
        # Tạo biểu đồ countplot
        sns.countplot(
            data=data, 
            x='sentiment', 
            palette='tab10', 
            ax=ax
        )
        # Thêm nhãn giá trị lên các cột
        for container in ax.containers: # type: ignore
            ax.bar_label(container)
        # Thiết lập tiêu đề và xoay nhãn trục X
        ax.set_title('Tần suất Positive/Negative trên tập dữ liệu', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # Tự động căn chỉnh layout
        plt.tight_layout()
        # Hiển thị biểu đồ
        st.pyplot(fig)


# Giao diện phần 'Thông tin về sản phẩm'
if info_options == 'Thông tin về sản phẩm':
    st.image('img/hasaki_logo.png', use_column_width=True)
    if st.session_state['uploaded_data'] is None:
        st.warning('Dataset chưa được tải lên')
    else:
        st.write('## Truy suất thông tin về một sản phẩm bất kỳ')
        data = st.session_state['uploaded_data']  # Lấy dữ liệu từ session_state
        # data = data.drop(data[data['ngay_binh_luan'] == '30/11/-0001'].index)
        # data['ngay_binh_luan'] = pd.to_datetime(data['ngay_binh_luan'], format='%Y-%m-%d')
        
        # Lấy sản phẩm
        data_info = data[['ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'gia_ban', 'processed_noi_dung_binh_luan', 'ngay_binh_luan']]
        random_products = data_info.drop_duplicates(subset='ma_san_pham')
        st.session_state.random_products = random_products
        
        # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
        if 'selected_ma_san_pham' not in st.session_state:
            st.session_state.selected_ma_san_pham = None # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
            
        # Theo cách cho người dùng chọn sản phẩm từ dropdown
        # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
        product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in st.session_state.random_products.iterrows()]

        # Tạo một dropdown với options là các tuple này
        selected_product = st.selectbox(
            'Chọn sản phẩm',
            options=product_options,
            format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
        )
        # # Display the selected product
        # st.write("Bạn đã chọn:", selected_product)
        
        # Cập nhật session_state dựa trên lựa chọn hiện tại
        st.session_state.selected_ma_san_pham = selected_product[1] # type: ignore

        if st.session_state.selected_ma_san_pham:
            st.write(f'ma_san_pham: {st.session_state.selected_ma_san_pham}')
            # Hiển thị thông tin sản phẩm được chọn
            selected_product = data[data['ma_san_pham'] == st.session_state.selected_ma_san_pham].sort_values(by='ngay_binh_luan', ascending=False)

            if not selected_product.empty:
                st.write('-'*3)
                st.write(f'#### {selected_product["ten_san_pham"].values[0]}')
                st.write(f'### {selected_product["diem_trung_binh"].values[0]} :star:', '{:,.0f}'.format(selected_product["gia_ban"].values[0]),'VNĐ')
                product_description = selected_product['mo_ta'].values[0]
                # truncated_description = ' '.join(product_description.split()[:100])
                # Tabs chính
                info_tabs = st.tabs(['Thông tin sản phẩm', 'Đánh giá từ khách hàng', 'Wordcloud'])
                with info_tabs[0]:
                    st.write(product_description.replace('THÔNG TIN SẢN PHẨM','').replace('Làm sao để phân biệt hàng có trộn hay không ?\nHàng trộn sẽ không thể xuất hoá đơn đỏ (VAT) 100% được do có hàng không nguồn gốc trong đó.\nTại Hasaki, 100% hàng bán ra sẽ được xuất hoá đơn đỏ cho dù khách hàng có lấy hay không. Nếu có nhu cầu lấy hoá đơn đỏ, quý khách vui lòng lấy trước 22h cùng ngày. Vì sau 22h, hệ thống Hasaki sẽ tự động xuất hết hoá đơn cho những hàng hoá mà khách hàng không đăng kí lấy hoá đơn.\nDo xuất được hoá đơn đỏ 100% nên đảm bảo 100% hàng tại Hasaki là hàng chính hãng có nguồn gốc rõ ràng.',''))
                with info_tabs[1]:
                    for i in range(len(selected_product["noi_dung_binh_luan"])):
                        st.write(f'{selected_product["ngay_binh_luan"].dt.strftime("%d-%m-%Y").values[i]}, {selected_product["ho_ten"].values[i]}, {selected_product["so_sao"].values[i]*":star:"}')
                        st.write(f'{selected_product["noi_dung_binh_luan"].values[i]}')
                        st.write('-'*3)
                with info_tabs[2]:
                    filtered_product = selected_product.groupby('ma_san_pham')['processed_noi_dung_binh_luan'].apply(' '.join).reset_index()
                    filtered_product.rename(columns={"processed_noi_dung_binh_luan": "merged_comments"}, inplace=True)
                    filtered_product['positive_words'] = filtered_product['merged_comments'].apply(lambda txt: ' '.join(tpr.find_words(txt, list_of_words=positive_words_lst)[1]))
                    filtered_product['negative_words'] = filtered_product['merged_comments'].apply(lambda txt: ' '.join(tpr.find_words(txt, list_of_words=negative_words_lst)[1]))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write('##### Wordcloud tích cực')

                        # Lấy giá trị đầu tiên từ mảng và xử lý các từ đặc biệt
                        positive_bowl = filtered_product['positive_words'].to_numpy()[0]
                        positive_bowl = tpr.process_special_word(positive_bowl)

                        # Đảm bảo positive_bowl là một chuỗi hợp lệ
                        if isinstance(positive_bowl, list):  # Nếu đầu ra là danh sách, nối các từ lại thành chuỗi
                            positive_bowl = ' '.join(positive_bowl)
                        elif not isinstance(positive_bowl, str):  # Nếu không phải chuỗi, chuyển đổi về chuỗi
                            positive_bowl = str(positive_bowl)

                        # Kiểm tra nếu không có chữ nào để tạo Wordcloud
                        if not positive_bowl.strip():  # .strip() để loại bỏ khoảng trắng
                            st.warning('Hiện tại chưa có chữ để trích xuất Wordcloud')
                        else:
                            # Tạo positive WordCloud
                            positive_wordcloud = wc(
                                width=800,
                                height=400,
                                max_words=25,
                                background_color='white',
                                colormap='viridis',
                                collocations=False
                            ).generate(positive_bowl)

                            # Hiển thị positive WordCloud trong Streamlit
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(positive_wordcloud, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
                    with col2:
                        st.write('##### Wordcloud tiêu cực')

                        # Lấy giá trị đầu tiên từ mảng và xử lý các từ đặc biệt
                        negative_bowl = filtered_product['negative_words'].to_numpy()[0]
                        negative_bowl = tpr.process_special_word(negative_bowl)

                        # Đảm bảo negative_bowl là một chuỗi hợp lệ
                        if isinstance(negative_bowl, list):  # Nếu đầu ra là danh sách, nối các từ lại thành chuỗi
                            negative_bowl = ' '.join(negative_bowl)
                        elif not isinstance(negative_bowl, str):  # Nếu không phải chuỗi, chuyển đổi về chuỗi
                            negative_bowl = str(negative_bowl)

                        # Kiểm tra nếu không có chữ nào để tạo Wordcloud
                        if not negative_bowl.strip():  # .strip() để loại bỏ khoảng trắng
                            st.warning('Hiện tại chưa có chữ để trích xuất Wordcloud')
                        else:
                            # Tạo negative WordCloud
                            negative_wordcloud = wc(
                                width=800,
                                height=400,
                                max_words=25,
                                background_color='white',
                                colormap='Oranges',
                                collocations=False
                            ).generate(negative_bowl)

                            # Hiển thị negative WordCloud trong Streamlit
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(negative_wordcloud, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
        else:
            st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")
        
if info_options == 'Dự báo thái độ cho dataset':
    st.image('img/hasaki_logo.png', use_column_width=True)
    if st.session_state['uploaded_data'] is None:
        st.warning('Dataset chưa được tải lên')
    else:
        st.write('## Dự báo thái độ bình luận trên một dataset')
        data = st.session_state['uploaded_data']  # Lấy dữ liệu từ session_state
        # Tabs chính
        model_tabs = st.tabs(['Dự đoán', 'Đánh giá kết quả'])
        with model_tabs[0]:
            if 'data' not in st.session_state:
                st.session_state['data'] = None  # Khởi tạo nếu chưa có dữ liệu
            if st.session_state['data'] is None:
                data['sentiment'] = data['so_sao'].apply(lambda txt: tpr.create_sentiment_col(target=txt, stars=4))
                data['label'] = label_encoder.fit_transform(data['sentiment'])
                st.write('Dữ liệu ban đầu:')
                st.dataframe(data[['so_sao', 'noi_dung_binh_luan']].head(5))
                st.dataframe(data[['so_sao', 'noi_dung_binh_luan']].tail(5))
                if st.button('Dự đoán thái độ các bình luận trong dataset'):
                    with st.spinner('Đang xử lý...'):
                        X_test = tpr.sentiment_predict(data, text_col_name='processed_noi_dung_binh_luan', trained_tfidf=proj1_tfidf_vectorizer)
                        y_pred = proj1_sentiment_lgr_model.predict(X_test)
                        data['label_pred'] = y_pred
                        data['Dự báo thái độ bình luận'] = data['label_pred'].apply(lambda txt: 'positive' if txt == 1 else 'negative')
                        st.success('Dự đoán hoàn tất!')
                        st.write('Kết quả phân loại:')
                        st.dataframe(data[['Dự báo thái độ bình luận', 'so_sao', 'noi_dung_binh_luan']])
                        # Lưu kết quả để đánh giá trong tab khác
                        st.session_state['data'] = data
                        st.session_state['y_pred'] = y_pred
            else:
                data = st.session_state['data']
                st.write('Dữ liệu ban đầu:')
                st.dataframe(data[['so_sao', 'noi_dung_binh_luan']].head(5))
                st.dataframe(data[['so_sao', 'noi_dung_binh_luan']].tail(5))
                st.success('Dự đoán hoàn tất!')
                st.write('Kết quả phân loại:')
                st.dataframe(data[['Dự báo thái độ bình luận', 'so_sao', 'noi_dung_binh_luan']])
                    
        with model_tabs[1]:
            st.header('Đánh giá mô hình')
            if 'data' in st.session_state and 'y_pred' in st.session_state:
                data = st.session_state['data']
                y_pred = st.session_state['y_pred']

                # Tính toán metrics
                metrics = evaluation.evaluate_model(data['label'], y_pred)
                st.subheader('Kết quả đánh giá:')
                st.json(metrics)

                # Hiển thị confusion matrix
                cm = confusion_matrix(data['label'], y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=proj1_sentiment_lgr_model.classes_)
                fig, ax = plt.subplots()
                disp.plot(ax=ax, cmap='Blues')
                st.pyplot(fig)
            else:
                st.warning('Vui lòng thực hiện dự đoán trước trong tab "Dự đoán".')

if info_options == 'Dự báo thái độ cho comment':
    st.image('img/hasaki_logo.png', use_column_width=True)
    if st.session_state['uploaded_data'] is None:
        st.warning('Dataset chưa được tải lên')
    else:
        st.write('## Dự báo thái độ một bình luận')
        data = st.session_state['uploaded_data']  # Lấy dữ liệu từ session_state
        st.write('Nhập một comment để kiểm tra sentiment')
        user_input = st.text_area('Nhập comment:')
        if st.button('Dự đoán'):
            if user_input == '':
                st.warning('Mời bạn nhập nội dung bình luận!')
            else:
                with st.spinner('Đang xử lý...'):
                    input_df = pd.DataFrame({'noi_dung_binh_luan': [user_input]})
                    input_df = tpr.txt_process_for_cols(input_df=input_df,
                                                        input_col_name='noi_dung_binh_luan',
                                                        emoji_dict=emoji_dict,
                                                        teen_dict=teen_dict,
                                                        translate_dict=translate_dict,
                                                        stopwords_lst=stopwords_lst,
                                                        groupby=False,
                                                        chunking=False)
                    X_test = tpr.sentiment_predict(input_df, text_col_name='noi_dung_binh_luan_special_words_remove_stopword', trained_tfidf=proj1_tfidf_vectorizer)
                    y_pred = proj1_sentiment_lgr_model.predict(X_test)[0]
                    input_df['label_pred'] = y_pred
                    input_df['Dự báo thái độ bình luận'] = input_df['label_pred'].apply(lambda txt: '**Positive**' if txt == 1 else '***Negative***')
                    st.success('Dự đoán hoàn tất!')
                    st.write('Nội dung bình luận có khả năng:', input_df['Dự báo thái độ bình luận'][0])
                    

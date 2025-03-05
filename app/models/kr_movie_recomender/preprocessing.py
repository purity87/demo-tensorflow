import re
import data_loader as dl


def preprocessing_df():
    # 데이터 가져오기
    df = dl.load_data()

    # NaN 값 제거
    df = df.dropna()

    # 중복 제거
    df = df.drop_duplicates(subset=['document'])

    # 한글 및 공백 제외하고 제거
    df['document'] = df['document'].apply(lambda x: re.sub(r"[^가-힣\s]", "", str(x)))

    # 데이터 확인
    print('---------------------------------- preprocessing_df START ----------------------------------')
    print(df.head())
    print('---------------------------------- preprocessing_df END ----------------------------------')
    return df

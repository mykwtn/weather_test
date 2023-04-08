#phase１
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# データの読み込み
df = pd.read_csv('weather_test8.csv', encoding='shift-jis')

# 目的変数と説明変数の設置
x = df.drop(['Class','Suggestion'], axis=1)
y = df['Class']

# ロジスティック回帰モデルの学習
clf = LogisticRegression()
clf.fit(x, y)

# Suggestionデータの読み込み
#suggestion_df = pd.read_csv('Suggestion_7.csv', encoding='shift-jis')
df2 = pd.read_csv('Suggestion_8.csv', header=None, names=['Suggestion'], encoding='shift-jis')


# 予測結果の取得

def get_suggestion_class(value_df):
    pred_probs = clf.predict_proba(value_df)
    pred_df = pd.DataFrame(pred_probs, columns=clf.classes_)
    class_label = pred_df.columns[np.argmax(pred_probs)]

    # Suggestionの取得
    suggestion = df2.iloc[int(class_label), 0]

    return suggestion, class_label



# 天気予測とSuggestionの表示
# メインパネル
st.title('Weather')
st.write('## Definition')

# サイドバー（入力画面）
#st.sidebar.header('Temp')
highest = st.slider('最高気温', min_value=0.0, max_value=40.0, value=20.0, step=0.1)
lowest = st.slider('最低気温', min_value=-10.0, max_value=40.0, value=10.0, step=0.1)
rainfall = st.slider('降水量', min_value=0.0, max_value=100.0, value=0.0, step=0.1)

# 最低気温が最高気温を上回っている場合にはエラーメッセージを表示する
if lowest > highest:
    st.error('最低気温が最高気温を上回っています。正しい値を入力してください。')
    st.stop()
    
# 入力値の値
value_df = pd.DataFrame([[highest, lowest, rainfall]], columns=['Highest', 'Lowest', 'rain'])
value_df.index = ['data']
st.write(value_df)


# 予測値の取得
suggestion = get_suggestion_class(value_df)


# 予測値の表示
st.write('## Suggestions')
st.write(suggestion)




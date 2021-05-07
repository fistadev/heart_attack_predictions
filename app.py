import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy import stats

# import category_encoders as ce
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing  # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection  # train_test_split
# accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import metrics
from sklearn import set_config
from sklearn.datasets import make_classification
# from feature_engine.selection import SmartCorrelatedSelection

from PIL import Image
import pickle


# Load Data

@st.cache
def load_data(filename=None):
    filename_default = './data/heart.csv'
    if not filename:
        filename = filename_default

    df = pd.read_csv(f"./{filename}")
    return df
    # return df, df.shape[0], df.shape[1], filename


data = load_data()

load_clf = pickle.load(open('./data/model.pkl', "rb"))


# def xgb_page_builder(data):
#     st.sidebar.header('Heart Attack Predictions')
#     st.sidebar.markdown('You can tune the parameters by siding')
#     cp = st.sidebar.slider('Select cp (default = 0)', 0, 1))
#     thale=st.sidebar.slider('Select Thalch (default = 150)',
#                               min_value = 50,
#                               max_value = 300,
#                               value = 150,
#                               step = 5)
#     slope=st.sidebar.slider('Select slope (default = 1)', 0, 1, 2)


############


# header=st.beta_container()
# team=st.beta_container()
# dataset=st.beta_container()
# footer=st.beta_container()


# with header:
#     st.title('Heart Attack Predictions')  # site title h1
#     st.markdown("""---""")
#     st.header('Machine Learning Project')
#     st.text(' ')
#     # image = Image.open('data/baby-yoda.jpg')
#     # st.image(image, caption="This is the way")
#     st.text(' ')
# with team:
#     # meet the team button
#     st.subheader('John Locke Team')
#     st.text(' ')
#     st.text(' ')
#     st.text(' ')
#     col1, col2, col3, col4=st.beta_columns(4)
#     with col1:
#         # image = Image.open('imgs/fabio.jpeg')
#         # st.image(image, caption="")
#         st.markdown(
#             '[Fabio Fistarol](https://github.com/fistadev)')
#     with col2:
#         # image = Image.open('imgs/madina.jpeg')
#         # st.image(image, caption="")
#         st.markdown(
#             '[Madina Zhenisbek](https://github.com/madinach)')
#     with col3:
#         # image = Image.open('imgs/alessio.jpg')
#         # st.image(image, caption="")
#         st.markdown(
#             '[Alessio Recchia](https://github.com/alessiorecchia)')
#     with col4:
#         # image = Image.open('imgs/shahid.jpg')
#         # st.image(image, caption="")
#         st.markdown(
#             '[Shahid Qureshi](https://github.com/shahidqureshi01)')

#     st.text(' ')
# st.text(' ')
# st.text(' ')
# st.markdown("""---""")
# st.text(' ')
# st.text(' ')
# image = Image.open('data/long_time_ago.jpg')
# st.image(image, caption="")

# Add audio
# audio_file = open('data/star_wars_theme_song.mp3', 'rb')
# audio_bytes = audio_file.read()
# st.audio(audio_bytes, format='audio/ogg')

##############

# st.header("Variables or features explanations:")
# st.markdown("""* age (Age in years)""")
# st.markdown("""* sex : (1 = male, 0 = female)""")
# st.markdown(
#     """* cp (Chest Pain Type): [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic]""")
# st.markdown("""* trestbps (Resting Blood Pressure in mm/hg )""")
# st.markdown("""* chol (Serum Cholesterol in mg/dl)""")
# st.markdown("""* fps (Fasting Blood Sugar > 120 mg/dl): [0 = no, 1 = yes]""")
# st.markdown(
#     """* restecg (Resting ECG): [0: normal, 1: having ST-T wave abnormality , 2: showing probable or definite left ventricular hypertrophy]""")
# st.markdown("""* thalach (maximum heart rate achieved)""")
# st.markdown("""* exang (Exercise Induced Angina): [1 = yes, 0 = no]""")
# st.markdown("""* oldpeak (ST depression induced by exercise relative to rest)""")
# st.markdown("""* slope (the slope of the peak exercise ST segment)""")
# st.markdown("""* ca [number of major vessels (0–3)]""")
# st.markdown(
#     """* thal : [1 = normal, 2 = fixed defect, 3 = reversible defect]""")
# st.markdown("""* target: [0 = disease, 1 = no disease]""")


###############


# with dataset:
#     st.header("")
#     # st.subheader("Galaxies")
#     st.markdown("")
#     st.markdown("")


########## k-means ###########
#     st.markdown("")
#     st.markdown("")
#     st.subheader("Pipeline")
#     st.text("Pipeline")
#     st.markdown("")

#     def drop_useless_columns(df, manual_drop_list = None):
#         if manual_drop_list:
#             return df.drop(columns = manual_drop_list), manual_drop_list

#         drop_list=[]
#         for col in df.columns:
#             if df[col].nunique() <= 1 or df[col].nunique() >= df.shape[0] * 0.95:
#                 drop_list.append(col)
#         return df.drop(columns = drop_list), drop_list

#     def datetime_processing(df):
#         if 'time' not in df.columns:
#             return df

#         df['weekday']=[x.weekday() for x in df.time]
#         df['hour']=[int(x.strftime('%H')) for x in df.time]
#         max_time=df.time.max()
#         min_time=df.time.min()
#         min_norm, max_norm=-1, 1
#         df['date']=(df.time - min_time) * (max_norm - min_norm) / \
#             (max_time - min_time) + min_norm
#         return df

#     def data_preprocessing(df, manual_drop_list = None):
#         """Return processed data and column list that need to be dropped."""

#         df, drop_list=drop_useless_columns(df, manual_drop_list)

#         df=datetime_processing(df)

#         df.C2=df.C1 + df.C3
#         df=df.drop(columns = ['time', 'X'])
#         df['sum_X']=df.iloc[:, 9:33].sum(axis = 1)

#         return df.replace({'status': {'Approved': 0, 'Declined': 1}}), drop_list


# ####### Logistic Regression ###############

# @ st.cache(suppress_st_warning = True)
# def logistic_train_metrics(df):
#     """Return metrics and model for Logistic Regression."""

#     X=df.drop(columns = ['status'])
#     Y=df.status

#     std_scaler=StandardScaler()
#     std_scaled_df=std_scaler.fit_transform(X)
#     std_scaled_df=pd.DataFrame(std_scaled_df, columns = X.columns)

#     X_train, X_test, y_train, y_test=train_test_split(
#         std_scaled_df, Y, random_state=0)

#     # Fit model
#     model_reg = LogisticRegression(max_iter=1000)
#     model_reg.fit(X_train.fillna(0), y_train)

#     # Make predictions for test data
#     y_pred = model_reg.predict(X_test.fillna(0))

#     # Evaluate predictions
#     accuracy_reg = accuracy_score(y_test, y_pred)
#     f1_reg = f1_score(y_test, y_pred)
#     roc_auc_reg = roc_auc_score(y_test, y_pred)
#     recall_reg = recall_score(y_test, y_pred)
#     precision_reg = precision_score(y_test, y_pred)

#     return accuracy_reg, f1_reg, roc_auc_reg, recall_reg, precision_reg, model_reg

##################################################################################

##### Plots #####


##################################################################################


def main():
    menu = ["Home", "Data Analysis", "Predictions"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        # st.subheader("Home")
        # to_do1 = st.checkbox("Web Scrapping ")
        # to_do2 = st.checkbox("Data Analysis")
        # to_do3 = st.checkbox("Data Prosessing")
        # to_do4 = st.checkbox("Data Visualization")
        # to_do5 = st.checkbox("About Dumblodore Team")
        # image = Image.open('imgs/dumbledore-on-strive.jpeg')
        # st.image(image, caption='Dumbledore')

        ####################################################
        header = st.beta_container()
        team = st.beta_container()
        activities = st.beta_container()
        github = st.beta_container()
        # dataset = st.beta_container()
        # conclusion = st.beta_container()
        # footer = st.beta_container()
        ####################################################
        with header:
            st.title('Heart Attack Predictions')  # site title h1
            st.markdown("""---""")
            st.header('Hippocratia - Machine Learning Project')
            st.text(' ')
            image = Image.open('./data/cardiacmonitor.png')
            st.image(image, caption="")

            with team:
                # meet the team button
                st.sidebar.subheader('John Locke Team')

                st.sidebar.markdown(
                    '[Fabio Fistarol](https://github.com/fistadev)')
                st.sidebar.markdown(
                    '[Madina Zhenisbek](https://github.com/madinach)')
                st.sidebar.markdown(
                    '[Alessio Recchia](https://github.com/alessiorecchia)')
                st.sidebar.markdown(
                    '[Shahid Qureshi](https://github.com/shahidqureshi01)')

                st.sidebar.text(' ')
                st.sidebar.text(' ')

            # with team:
            #     # meet the team button
            #     st.subheader('John Locke Team')
            #     st.text(' ')
            #     st.text(' ')
            #     col1, col2, col3, col4 = st.beta_columns(4)
            #     with col1:
            #         # image = Image.open('imgs/fabio.jpeg')
            #         # st.image(image, caption="")
            #         st.markdown(
            #             '[Fabio Fistarol](https://github.com/fistadev)')
            #     with col2:
            #         # image = Image.open('imgs/madina.jpeg')
            #         # st.image(image, caption="")
            #         st.markdown(
            #             '[Madina Zhenisbek](https://github.com/madinach)')
            #     with col3:
            #         # image = Image.open('imgs/alessio.jpg')
            #         # st.image(image, caption="")
            #         st.markdown(
            #             '[Alessio Recchia](https://github.com/alessiorecchia)')
            #     with col4:
            #         # image = Image.open('imgs/shahid.jpg')
            #         # st.image(image, caption="")
            #         st.markdown(
            #             '[Shahid Qureshi](https://github.com/shahidqureshi01)')

            #     st.text(' ')
            #     st.text(' ')

        # with activities:
        #     # activities section:
        #     st.header('Activities')
        #     st.markdown('* Webscraping')
        #     st.markdown('* Data Visualisation')
        #     st.markdown('* Data Analysis ')
        #     st.markdown('* Analising Business Scenario')
        #     st.text(' ')

        with github:
            # github section:
            st.header('GitHub / Instructions')
            st.markdown(
                'Check the instruction [here](https://github.com/fistadev/heart_attack_predictions)')
            st.text(' ')


##########################################################################
    elif choice == "Data Analysis":
        st.subheader("Data Analysis")
        # data = load_data('raw')
        header = st.beta_container()
        dataset = st.beta_container()

        with dataset:
            st.title("Data Analysis")

            #### Data Correlation ####
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.text('Data Correlation ')
            sns.set(style="white")
            plt.rcParams['figure.figsize'] = (15, 10)
            sns.heatmap(data.corr(), annot=True, linewidths=.5, cmap="Blues")
            plt.title('Corelation Between Variables', fontsize=30)
            plt.show()
            st.pyplot()

            #### Box Plot #####
            st.text('Outlier Detection ')
            fig = plt.figure(figsize=(15, 10))
            sns.boxplot(data=data)
            st.pyplot(fig)

            # book_data = load_data("raw")
            # st.write(book_data.head(10))
            ##### Sidebar #####

        #     st.sidebar.markdown("Which Type of Graph do want?")
        # np.select = st.sidebar.selectbox(
        #     "Graph type", ["Number of rating and Awards", "The top 15 best author", "Average Rating Analysis", "The published books by year"], key='1')
        # # np.select = st.sidebar.selectbox(
        # #     "Graph type", ["Number of rating and Awards", "The top 15 best author", "Award and Year", "Average Rating Analysis", "The published books by year"], key='1')
        # if np.select == "Number of rating and Awards":
        #     # st.markdown(
        #     #     '- Create a 2D scatterplot with pages on the x-axis and num_ratings on the y-axis.')
        #     st.text(" ")
        #     scatter_2D_plot(data)
        #     st.pyplot()
        # ################################################
        #     # Bar Charts
        # if np.select == "The top 15 best author":
        #     best_book(data)
        #     st.pyplot()
        #     # group_bar_chart(data)
        #     # st.pyplot()

        # ################################################
        # if np.select == "Average Rating Analysis":
        #     norm_functions(data)
        #     st.pyplot()

        # ################################################
        # if np.select == "The published books by year":
        #     group_bar_chart(data)

            st.text(' ')
            st.header("Variables or features explanations:")
            st.markdown("""* age (Age in years)""")
            st.markdown("""* sex : (1 = male, 0 = female)""")
            st.markdown(
                """* cp (Chest Pain Type): [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic]""")
            st.markdown("""* trestbps (Resting Blood Pressure in mm/hg )""")
            st.markdown("""* chol (Serum Cholesterol in mg/dl)""")
            st.markdown(
                """* fps (Fasting Blood Sugar > 120 mg/dl): [0 = no, 1 = yes]""")
            st.markdown(
                """* restecg (Resting ECG): [0: normal, 1: having ST-T wave abnormality , 2: showing probable or definite left ventricular hypertrophy]""")
            st.markdown("""* thalach (maximum heart rate achieved)""")
            st.markdown(
                """* exang (Exercise Induced Angina): [1 = yes, 0 = no]""")
            st.markdown(
                """* oldpeak (ST depression induced by exercise relative to rest)""")
            st.markdown(
                """* slope (the slope of the peak exercise ST segment)""")
            st.markdown("""* ca [number of major vessels (0–3)]""")
            st.markdown(
                """* thal : [1 = normal, 2 = fixed defect, 3 = reversible defect]""")
            st.markdown("""* target: [0 = disease, 1 = no disease]""")

    elif choice == "ML":
        # st.subheader("Business Analysis")

        # with conclusion:
        #     st.title('Plots')
        #     st.text(' ')
        #     st.header("Variables or features explanations:")
        #     st.markdown("""* age (Age in years)""")
        #     st.markdown("""* sex : (1 = male, 0 = female)""")
        #     st.markdown(
        #         """* cp (Chest Pain Type): [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic]""")
        #     st.markdown("""* trestbps (Resting Blood Pressure in mm/hg )""")
        #     st.markdown("""* chol (Serum Cholesterol in mg/dl)""")
        #     st.markdown(
        #         """* fps (Fasting Blood Sugar > 120 mg/dl): [0 = no, 1 = yes]""")
        #     st.markdown(
        #         """* restecg (Resting ECG): [0: normal, 1: having ST-T wave abnormality , 2: showing probable or definite left ventricular hypertrophy]""")
        #     st.markdown("""* thalach (maximum heart rate achieved)""")
        #     st.markdown(
        #         """* exang (Exercise Induced Angina): [1 = yes, 0 = no]""")
        #     st.markdown(
        #         """* oldpeak (ST depression induced by exercise relative to rest)""")
        #     st.markdown(
        #         """* slope (the slope of the peak exercise ST segment)""")
        #     st.markdown("""* ca [number of major vessels (0–3)]""")
        #     st.markdown(
        #         """* thal : [1 = normal, 2 = fixed defect, 3 = reversible defect]""")
        #     st.markdown("""* target: [0 = disease, 1 = no disease]""")

        with footer:
            # Footer
            st.markdown("""---""")
            st.markdown("Heart Attack Predictions - Machine Learning Project")
            st.markdown("")
            st.markdown(
                "If you have any questions, checkout our [documentation](https://github.com/fistadev/starwars_data_project) ")
            st.text(' ')

        ############################################################################################################################
    else:
        st.header("Predictions")

        def xgb_page_builder(data):
            st.sidebar.header('Heart Attack Predictions')
            st.sidebar.markdown('You can tune the parameters by siding')
            st.sidebar.text_input("What's your age?")
            max_depth = st.sidebar.slider(
                'Select max_depth (default = 30)', 3, 30, 30)
            eta = st.sidebar.slider(
                'Select learning rate (divided by 10) (default = 0.1)', 0.01, 1.0, 1.0)
            min_child_weight = st.sidebar.slider(
                'Select min_child_weight (default = 0.3)', 0.1, 3.0, 0.3)
            subsample = st.sidebar.slider(
                'Select subsample (default = 0.75)', 0.5, 1.0, 0.75)
            colsample_bylevel = st.sidebar.slider(
                'Select colsample_bylevel (default = 0.5)', 0.5, 1.0, 0.5)

        st.write(xgb_page_builder(data))

        st.text(' ')
        st.markdown('Model selection')
        st.text(' ')
        image = Image.open('./data/model-selection.png')
        st.image(image, caption="")
        st.text(' ')

        st.text(' ')
        st.markdown('Selecting the best model with KFold')
        st.text(' ')
        image = Image.open('./data/kfold.png')
        st.image(image, caption="")
        st.text(' ')

        # [
        # 54,# 'age',
        # 1, # 'sex',
        # # 'cp',
        # 131,# 'trestbps',
        # 246,# 'chol',
        # 0.148,# 'fbs',
        # 5280# 'restecg',
        # 149.6, # 'thalach',
        # 0.326, # 'exang',
        # 1.0396# 'oldpeak',
        # #  'slope',
        # 0.7293, # 'ca',
        # 2.313,# 'thal',
        # # 'target'
        # ]

# results_ord = results.sort_values(
#     by=['Accuracy'], ascending=False, ignore_index=True)
# results_ord.index += 1
# results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'],
#                       vmin=0, vmax=100, color='#5fba7d')
        set_config(display='diagram')
        st.write(load_clf)

# data = load_data("clean")
# the all graphic functions
#  Scatter plots

# def scatter_2D_plot(data):
#     st.markdown("")
#     st.markdown("")
#     st.subheader("Comparison between Number of rating and Awards")
#     st.markdown("")
#     st.markdown("")
#     size_b = data['award']**2*12
#     colors = np.random.rand(data.shape[0])
#     sns.scatterplot(data['num_pages'], data['num_rating'],
#                     s=size_b, c=colors, alpha=0.5, legend=True)

# def group_bar_chart(data):
#     st.markdown("")
#     st.markdown("")
#     st.subheader("Books Published by Year ")
#     st.markdown("")
#     st.markdown("")
#     tmp = data.groupby("original_publish_year")[
#         "award"].mean().sort_values()
#     st.bar_chart(tmp)

# def norm_functions(data):
#     st.markdown("")
#     st.markdown("")
#     st.subheader(
#         "Average Rating Analysis")
#     st.markdown("")
#     sns.histplot(data, x="avg_rating", color="green",
#                  label="Before Normalization", kde=True)
#     sns.histplot(data, x="minmax_norm_ratings", color="skyblue",
#                  label="Min-Max Normalization", kde=True)
#     sns.histplot(data, x="mean_norm_ratings", color="red",
#                  label="Mean Normalization", kde=True)
#     x1 = data["minmax_norm_ratings"]
#     x2 = data["mean_norm_ratings"]
#     x3 = data["avg_rating"]
#     hist_data = [x1, x2, x3]
#     group_labels = ['Min-Max Normalization',
#                     'Mean Normalization', 'Before Normalization Avarge rate']

#     fig = ff.create_distplot(hist_data, group_labels, bin_size=0.1)
#     st.plotly_chart(fig, use_container_width=True)

# def best_book(df):
#     st.markdown("")
#     st.markdown("")
#     st.subheader(
#         "The top 15 Best Author")
#     st.markdown("")
#     st.markdown("")
#     df = data.sort_values(by='award', ascending=False).reset_index(
#         drop=True).head(15)

#     sns.barplot(x="award", y="author", data=df,
#                 label='The best author who has more awards')

########################################################################################################


main()

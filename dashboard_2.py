#from typing import final
#from unicodedata import category
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

#Pembuatan fungsi untuk pivot tabel
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_approved_at').agg({
        "Unique_Value": "nunique",
        "price": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "Unique_Value": "order_count",
        "price": "revenue"
    }, inplace=True)
    return daily_orders_df

def create_monthly_order_df(df):
    monthly_revenue = df.resample(rule='M', on='order_approved_at').agg({
    'Unique_Value' : 'nunique',
    'price' : 'sum'
    })
    monthly_revenue.index = monthly_revenue.index.strftime('%Y-%m')
    monthly_revenue= monthly_revenue.reset_index()
    monthly_revenue.rename(columns={
    'Unique_Value' : 'order_count',
    'price' : 'revenue'}, inplace=True)
    return monthly_revenue

def create_gross_rating1(df) :
    grossing_rating = df.groupby(by='product_category_name_english').agg({
    'review_score' : 'mean',
    'Unique_Value' : 'nunique',
    'price' : 'sum'}).sort_values(by='Unique_Value', ascending=False).reset_index()
    grossing_rating.columns=['product_category', 'review_score', 'total_order', 'total_revenue']
    return grossing_rating

def create_gross_rating_2(df) :
    grossing_rating2 = df.groupby(by='product_category_name_english').agg({
    'review_score' : 'mean',
    'Unique_Value' : 'nunique',
    'price' : 'mean',}).sort_values(by='price', ascending=False).reset_index()
    grossing_rating2.columns=['product_category', 'review_score', 'total_order', 'average_price'] 
    return grossing_rating2

def create_gross_rating_3(df) :
    grossing_rating3 = df.groupby(by='product_category_name_english').agg({
    'review_score' : 'mean',
    'Unique_Value' : 'nunique',
    'price' : 'sum',}).sort_values(by='price', ascending=False).reset_index()
    grossing_rating3.columns=['product_category', 'review_score', 'total_order', 'total_revenue']
    return grossing_rating3

def create_category_city(df) :
    category_city = df.groupby(by='customer_city').agg({
        'Unique_Value' : 'nunique',
        'product_category_name_english' : 'nunique',
        'review_score' : 'mean',
        'price' : 'sum'
    }).sort_values(by='Unique_Value', ascending=False).reset_index()
    category_city.columns=['customer_city', 'total_order', 'product_category_count', 'review_score', 'total_revenue']
    return category_city
#RFM Analysis
def create_recency(df) :
    customer_recency= df.groupby(by='customer_id', as_index=False)['order_approved_at'].max()
    customer_recency.columns=['customer_id', 'last_purchased_time (from order)']
    recent_date= customer_recency['last_purchased_time (from order)'].max()
    customer_recency['recency']=customer_recency['last_purchased_time (from order)'].apply(
        lambda x: (recent_date - x).days)
    return customer_recency
def create_freqency(df) :
    customer_frequency= df.drop_duplicates().groupby(
    by=['customer_id'], as_index=False)['order_approved_at'].count()
    customer_frequency.columns=['customer_id', 'frequency']
    return customer_frequency
def create_monetary(df) :
    customer_monetary =df.groupby(by='customer_id', as_index=False)['price'].sum()
    customer_monetary.columns=['customer_id', 'monetary'] 
    return customer_monetary
    
#file yang diperlukan
#final_customer_order= pd.read_csv("Downloads/hello-world/submission/dashboard/final_customer_order.csv")
url="https://github.com/AhmadZakkiZainalAbidin/firstapp/raw/main/final_customer_order.csv"
final_customer_order= pd.read_csv(url, index_col=0)

datetime_columns =['order_approved_at']
final_customer_order.sort_values(by='order_approved_at', inplace=True)
final_customer_order.reset_index(inplace=True)

for column in datetime_columns :
    final_customer_order[column] = pd.to_datetime(final_customer_order[column])

#Membuat komponen Filter
min_date = final_customer_order['order_approved_at'].min()
max_date = final_customer_order['order_approved_at'].max()

with st.sidebar :  
    st.header("NoPicture:blush:")
    st.subheader(":sparkles:Submission:sparkles:")
    start_date, end_date = st.date_input(
        label='Time Range Setting', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
    st.write(":round_pushpin: Writer Note :")
    st.markdown("This is my first experience to be datascientist with working in this project, I am so excited about to do it. Even so this project is not perfect so be free to send your feedback to me. Btw enjoy this project :heart:")
    st.write()
    text = st.text_area('Feedback')
    st.write('Feedback: ', text)

main_df = final_customer_order[(final_customer_order['order_approved_at'] >= str(start_date)) & 
                (final_customer_order['order_approved_at'] <= str(end_date))]

daily_orders_df = create_daily_orders_df(main_df)
monthly_revenue =create_monthly_order_df(main_df)
grossing_rating = create_gross_rating1(main_df)
grossing_rating2 = create_gross_rating_2(main_df)
grossing_rating3 = create_gross_rating_3(main_df)
category_city = create_category_city(main_df)
#RFM Analysis memiliki All time input
customer_recency= create_recency(final_customer_order)
customer_frequency= create_freqency(final_customer_order)
customer_monetary = create_monetary(final_customer_order)


st.header(':stars: Dasboard From Ecommerce Dataset :stars:')

st.subheader(':round_pushpin: Total Orders and Revenue')

tab1, tab2, tab3 =st.tabs(['By Month', 'By Day', 'By City'])
with tab1 :
    col1, col2= st.columns(2)
    with col1:
        total_orders = monthly_revenue.order_count.sum()
        st.metric("Total Orders", value=total_orders)
    with col2:
        mean_monthly_order = monthly_revenue.order_count.mean()
        st.metric('Average Orders', value= round(mean_monthly_order))
    with st.container():
        col1, col2= st.columns(2)
        with col1:
            mean_monthly_revenue = monthly_revenue.revenue.mean()
            st.metric('Mean Revenue', value= ("${:,}".format(round(mean_monthly_revenue, 2))))
        with col2:
            total_revenue = monthly_revenue.revenue.sum()
            st.metric("Total Revenue", value=("${:,}".format(round(total_revenue, 2))))
    with st.container():
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_title(f'Number of Orders from E-Commers Public Dataset per Month \n for {str(start_date)} to {str(end_date)} Period', loc='center', fontsize=20)
        ax.plot(
            monthly_revenue['order_approved_at'],
            monthly_revenue["order_count"],
            marker='o', 
            ms=10, mfc='r', linewidth=3, color='#72BCD4')
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(rotation=45)
        st.pyplot(fig)
with tab2 :
    col1, col2 = st.columns(2)
    with col1:
        total_orders = daily_orders_df.order_count.sum()
        st.metric("Total Orders", value=total_orders)
    with col2:
        mean_daily_order = daily_orders_df.order_count.mean()
        st.metric('Average Orders', value= round(mean_daily_order))
    with st.container():
        col1, col2 = st.columns(2)
        with col1 :
            mean_daily_revenue= daily_orders_df.revenue.mean()
            st.metric('Mean Revenue', value=("${:,}".format(round(mean_daily_revenue, 2)))) 
        with col2: 
            total_revenue = daily_orders_df.revenue.sum() 
            st.metric("Total Revenue", value=("${:,}".format(round(total_revenue, 2))))
    with st.container():
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_title(f'Number of Orders from E-Commers Public Dataset per Daily \n for {str(start_date)} to {str(end_date)} Period', loc='center', fontsize=20)
        ax.plot(
            daily_orders_df['order_approved_at'],
            daily_orders_df["order_count"],
            marker='o', mfc='r', linewidth=2, color='#72BCD4')
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=15)
        st.pyplot(fig)
with tab3 :
    st.markdown(f"<h4 style='text-align: center; color: black;'>Hingest Total Number of Orders and Revenues by City for {str(start_date)} to {str(end_date)} Period </h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        max_order_city = category_city._get_value(0, 'customer_city')
        st.metric("City with Hingest Order", value=str(max_order_city))
    with col2:
        max_order_by_city = category_city.total_order.max()
        st.metric("Total Orders", value=max_order_by_city)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            max_revenue_city = category_city._get_value(0, 'customer_city') 
            st.metric("City with Hingest Revenue", value=str(max_revenue_city))
        with col2:
            max_revenue_by_city= category_city.total_revenue.max()
            st.metric('Total Revenue', value=("${:,}".format(round(max_revenue_by_city, 2))))
    with st.container():
        fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(20,9))
        colors1 = ["#FFD700", "#FFEEBC", "#FFEEBC", "#FFEEBC", "#FFEEBC", "#FFEEBC", "#FFEEBC", "#FFEEBC", "#FFEEBC", "#FFEEBC"]
        colors2 = ['#DC6DDC', "#ADD8E6", "#ADD8E6", "#ADD8E6", "#ADD8E6", "#ADD8E6", "#ADD8E6", "#ADD8E6", "#ADD8E6", "#ADD8E6"]
        #Tabel baris ke 1 kolom ke 1
        ax[0]=sns.barplot(x='total_order', y='customer_city', data=category_city.head(10), palette=colors1, ax=ax[0])
        for index, row in category_city.head(10).iterrows():
            ax[0].text(row.name, row.name, (f'           {round(row.total_order, 2)} Orders'), color='Black', fontsize=10, fontstyle='oblique')
            ax[0].set_ylabel(None)
            ax[0].set_xlabel('Number of Orders by City', fontsize=12)
            ax[0].tick_params(axis='y', rotation=0)
            ax[0].set_title(f'Hingest Number of Orders by City \n for {str(start_date)} to {str(end_date)} Period', fontsize=13)
            ax[0].tick_params(axis='y', labelsize=11)
            #Tabel baris ke 1 kolom ke 2
        ax[1]=sns.barplot(x='total_revenue', y='customer_city', data=category_city.sort_values(by='total_revenue', ascending=False).head(10), palette=colors2, ax=ax[1])
        for index, row in category_city.sort_values(by='total_revenue', ascending=False).head(10).reset_index().iterrows():
            ax[1].text(row.name, row.name, ("               ${:,}".format(round(row.total_revenue, 2))), color='Black', fontsize=10, fontstyle='oblique')
            ax[1].set_ylabel(None)
            ax[1].set_xlabel('Total Revenue in Dollars', fontsize=12)
            ax[1].tick_params(axis='y', rotation=0)
            ax[1].set_title(f'Hingest Total Revenues by City \n for {str(start_date)} to {str(end_date)} Period', fontsize=13)
            ax[1].tick_params(axis='y', labelsize=11)
        st.pyplot(fig)

st.subheader(':round_pushpin: Product Performance & Information')

tab1, tab2, tab3 = st.tabs(['By Total Order', 'By Total Revenue', 'By Average Price to Order Count'])
with tab1 :
    st.markdown(f"<h5 style='text-align: center; color: black;'>Hingest Grossing Product Category by Number of Orders with Reviews Score for {str(start_date)} to {str(end_date)} Period </h5>", unsafe_allow_html=True)
    col1, col2, col3= st.columns(3)
    with col1:
        hingest_grosproduct_name = grossing_rating._get_value(0, 'product_category')
        st.metric("Top Gross Product Category", value=str(hingest_grosproduct_name))
    with col2:
        hingest_grosproduct_count = grossing_rating._get_value(0, 'total_order')
        st.metric('Order count for Top Product', value= round(hingest_grosproduct_count))
    with col3:
        hingest_grosproduct_review = round(grossing_rating._get_value(0, 'review_score'), 3)
        st.metric('Review Score for Top Order product', value=str(hingest_grosproduct_review))
    with st.container():
        col1, col2, col3= st.columns(3)
        with col1:
            lowest_grosproduct_name = grossing_rating.sort_values(by='total_order', ascending=True).reset_index()._get_value(0, 'product_category')
            st.metric("Lowest Gross Product Category", value=str(lowest_grosproduct_name))
        with col2:
            lowest_grosproduct_count = grossing_rating.sort_values(by='total_order', ascending=True).reset_index()._get_value(0, 'total_order')
            st.metric('Order count for Lowest Product', value= round(lowest_grosproduct_count))
        with col3:
            lowest_grosproduct_review = round(grossing_rating.sort_values(by='total_order', ascending=True).reset_index()._get_value(0, 'review_score'), 3)
            st.metric('Review Score for Lowest Order product', value=str(lowest_grosproduct_review))

    with st.container() :
        fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(20,10))
        colors1 = ["#FFD700", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
        colors2 = ["#FF0000", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
        
        ax[0]=sns.barplot(x='total_order', y='product_category', data=grossing_rating.head(), palette=colors1, ax=ax[0])
        for index, row in grossing_rating.head().iterrows():
            ax[0].text(row.name, row.name, (f'           {round(row.review_score, 2)} Reviews Score'), color='Black', fontsize=12)
        ax[0].set_ylabel(None)
        ax[0].set_xlabel('Number of Orders Item', fontsize=16)
        ax[0].tick_params(axis='y', rotation=0)
        ax[0].set_title(f'Hingest Grossing Product Category by \n Number of Orders with Reviews Score \n for {str(start_date)} to {str(end_date)} Period', fontsize=18)
        ax[0].tick_params(axis='y', labelsize=11)
        
        ax[1]=sns.barplot(x='total_order', y='product_category', data=grossing_rating.sort_values(by='total_order', ascending=True).head(), palette=colors2, ax=ax[1])
        for index, row in grossing_rating.sort_values(by='total_order', ascending=True).head().reset_index().iterrows():
            ax[1].text(row.name, row.name, (f'{round(row.review_score, 2)} Reviews Score'), ha='right', color='Black', fontsize=12)
        ax[1].set_ylabel(None)
        ax[1].set_xlabel('Number of Orders Item', fontsize=16)
        ax[1].invert_xaxis()
        ax[1].yaxis.set_label_position('right')
        ax[1].yaxis.tick_right()
        ax[1].tick_params(axis='y', rotation=0)
        ax[1].set_title(f'Lowest Grossing Product Category by \n Number of Order with Reviews Score \n for {str(start_date)} to {str(end_date)} Period', fontsize=18)
        ax[1].tick_params(axis='y', labelsize=11)
        st.pyplot(fig)
with tab2 :
    st.markdown(f"<h5 style='text-align: center; color: black;'>Hingest Grossing Product Category by Revenue with Reviews Score for {str(start_date)} to {str(end_date)} Period </h5>", unsafe_allow_html=True)
    col1, col2, col3= st.columns(3)
    with col1:
        hingest_revenue_name = grossing_rating3._get_value(0, 'product_category')
        st.metric("Top Revenue Product Category", value=str(hingest_revenue_name))
    with col2:
        hingest_revenue_count = grossing_rating3._get_value(0, 'total_revenue') 
        st.metric('Top Revenue Product Category', value= ("${:,}".format(round(hingest_revenue_count, 2))))
    with col3:
        hingest_revenue_review = round(grossing_rating3._get_value(0, 'review_score'), 3)
        st.metric('Review Score for Top Revenue', value=str(hingest_revenue_review))
    with st.container():
        col1, col2, col3= st.columns(3)
        with col1:
            lowest_revenue_name = grossing_rating3.sort_values(by='total_revenue', ascending=True).reset_index()._get_value(0, 'product_category')
            st.metric("Lowest Revenue Product Category", value=str(lowest_revenue_name))
        with col2:
            lowest_revenue_count = grossing_rating3.sort_values(by='total_revenue', ascending=True).reset_index()._get_value(0, 'total_revenue')
            st.metric('Lowest Total Revenue Product', value= ("${:,}".format(round(lowest_revenue_count, 2))))
        with col3:
            lowest_revenue_review = round(grossing_rating3.sort_values(by='total_revenue', ascending=True).reset_index()._get_value(0, 'review_score'), 3)
            st.metric('Review Score for Lowest Revenue', value=str(lowest_revenue_review))
    with st.container() :
        fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(20,10))
        colors3 = ['#FFA500', "#BBA8FF", "#BBA8FF", "#BBA8FF", "#BBA8FF"]
        colors4 = ["#FF7F7F", "#BBA8FF", "#BBA8FF", "#BBA8FF", "#BBA8FF"]
        ax[0]=sns.barplot(x='total_revenue', y='product_category', data=grossing_rating3.head(), palette=colors3, ax=ax[0])
        for index, row in grossing_rating3.head().iterrows():
            ax[0].text(row.name, row.name, (f'           {round(row.review_score, 2)} Reviews Score'), color='Black', fontsize=12)
        ax[0].set_ylabel(None)
        ax[0].set_xlabel('Total Revenue in Million Dollars', fontsize=16)
        ax[0].tick_params(axis='y', rotation=0)
        ax[0].set_title(f'Hingest Gross Product Revenue \n with Reviews Score \n for {str(start_date)} to {str(end_date)} Period', fontsize=18)
        ax[0].tick_params(axis='y', labelsize=11)
        
        ax[1]=sns.barplot(x='total_revenue', y='product_category', data=grossing_rating3.sort_values(by='total_revenue', ascending=True).head(), palette=colors4, ax=ax[1])
        for index, row in grossing_rating3.sort_values(by='total_revenue', ascending=True).head().reset_index().iterrows():
            ax[1].text(row.name, row.name, (f'{round(row.review_score, 2)} Reviews Score    '), ha='right', color='Black', fontsize=12)
        ax[1].set_ylabel(None)
        ax[1].set_xlabel(None)
        ax[1].set_xlabel('Total Revenue in Dollars', fontsize=16)
        ax[1].invert_xaxis()
        ax[1].yaxis.set_label_position('right')
        ax[1].yaxis.tick_right()
        ax[1].tick_params(axis='y', rotation=0)
        ax[1].set_title(f'Lowest Gross Product Revenue \n with Reviews Score \n for {str(start_date)} to {str(end_date)} Period', fontsize=18)
        ax[1].tick_params(axis='y', labelsize=11)
        st.pyplot(fig)
with tab3 :
    st.markdown(f"<h5 style='text-align: center; color: black;'>Expensive and Ceapest Product Category by Average Price with Reviews Score for {str(start_date)} to {str(end_date)} Period </h5>", unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        expensive_category_name = grossing_rating2._get_value(0, 'product_category')
        st.metric("Expensive Product Category", value=str(hingest_revenue_name))
    with col2:
        expensive_product_price = grossing_rating2._get_value(0, 'average_price') 
        st.metric('Expensive Product Category Mean Price', value= ("${:,}".format(round(expensive_product_price, 2))))
    with st.container():
        col1, col2= st.columns(2)
        with col1:
            cheapest_product_name = grossing_rating2.sort_values(by='average_price', ascending=True).reset_index()._get_value(0, 'product_category')
            st.metric("Lowest Revenue Product Category", value=str(cheapest_product_name))
        with col2:
            cheapest_product_price = grossing_rating2.sort_values(by='average_price', ascending=True).reset_index()._get_value(0, 'average_price')
            st.metric('Lowest Total Revenue Product', value= ("${:,}".format(round(cheapest_product_price, 2))))
    with st.container() :
        fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(20,10))
        colors1 = ["#FFD700", "#ADD8E6", "#ADD8E6", "#ADD8E6", "#ADD8E6"]
        colors2 = ["#FF0000", "#ADD8E6", "#ADD8E6", "#ADD8E6", "#ADD8E6"]
        ax[0]=sns.barplot(x='average_price', y='product_category', data=grossing_rating2.head(), palette=colors1, ax=ax[0])
        for index, row in grossing_rating2.head().iterrows():
            ax[0].text(row.name, row.name, (f'           {round(row.total_order, 2)} Orders'), color='Black', fontsize=12)
        ax[0].set_ylabel(None)
        ax[0].set_xlabel('Average Price in Dollars', fontsize=14)
        ax[0].tick_params(axis='y', rotation=0)
        ax[0].set_title('Hingest Average Prices Product  \n with Number of Order', fontsize=20)
        ax[0].tick_params(axis='y', labelsize=11)
        
        ax[1]=sns.barplot(x='average_price', y='product_category', data=grossing_rating2.sort_values(by='average_price', ascending=True).head(), palette=colors2, ax=ax[1])
        for index, row in grossing_rating2.sort_values(by='average_price', ascending=True).head().reset_index().iterrows():
            ax[1].text(row.name, row.name, (f'{round(row.total_order, 2)} Orders    '), ha='right', color='Black', fontsize=12)
        ax[1].set_ylabel(None)
        ax[1].set_xlabel(None)
        ax[1].set_xlabel('Average Price in Dollars', fontsize=14)
        ax[1].invert_xaxis()
        ax[1].yaxis.set_label_position('right')
        ax[1].yaxis.tick_right()
        ax[1].tick_params(axis='y', rotation=0)
        ax[1].set_title('Lowest Average Prices Product by \n with Number of Order', fontsize=20)
        ax[1].tick_params(axis='y', labelsize=11)
        st.pyplot(fig)

st.subheader(':round_pushpin: Customer Category Analysis With RFM (All Time)')
with st.container() :
    st.markdown(f"<h5 style='text-align: center; color: black;'>Recency, Frequency, and Monetary Info for Top Customer Period </h5>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1 :
        mean_recency = round(customer_recency.recency.mean(), 1)
        st.metric('Average Recency (days)', value=mean_recency)
    with col2 :
        mean_frequency = round(customer_frequency.frequency.mean(), 1)
        st.metric('Average Frequency', value=mean_frequency)
    with col3 :
        mean_monetary = round(customer_monetary.monetary.mean(), 2)
        st.metric("Average Monetary", value="${:,}".format(round(mean_monetary, 2)))
with st.container() :
    st.markdown(f"<h5 style='text-align: center; color: black;'>Analysis of All Customer Category that use E-commers (2016-2018) based on RFM Score </h5>", unsafe_allow_html=True)    
    #Pivot RFM untuk seluruh dataset customer
    customer_rf = customer_recency.merge(customer_frequency, on='customer_id')
    customer_rfm = customer_rf.merge(customer_monetary, on='customer_id').drop(columns='last_purchased_time (from order)')
    customer_rfm['R_rank'] = customer_rfm['recency'].rank(ascending=False)
    customer_rfm['F_rank'] = customer_rfm['frequency'].rank(ascending=True)
    customer_rfm['M_rank'] = customer_rfm['monetary'].rank(ascending=True)
    #Melakukan normalisasi terhadap data peringkat customer_id
    customer_rfm['R_rank_norm'] = (customer_rfm['R_rank']/customer_rfm['R_rank'].max())*100
    customer_rfm['F_rank_norm'] = (customer_rfm['F_rank']/customer_rfm['F_rank'].max())*100
    customer_rfm['M_rank_norm'] = (customer_rfm['F_rank']/customer_rfm['M_rank'].max())*100
    customer_rfm.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)
    #Melakukan perhitungan RFM_score untuk seluruh customer_id
    customer_rfm['RFM_Score'] = (0.15*customer_rfm['R_rank_norm']+0.28 * \
                                 customer_rfm['F_rank_norm']+0.57*customer_rfm['M_rank_norm'])
    customer_rfm['RFM_Score'] *= 0.05
    customer_rfm = customer_rfm.round(2)
    #Melakukan kategori customer berdasarkan nilai RFM nya
    customer_rfm["Customer_segment"] = np.where(
        customer_rfm['RFM_Score'] > 4.5, "Top Customers",
        (np.where(customer_rfm['RFM_Score'] > 4,"High value Customer",
                  (np.where(customer_rfm['RFM_Score'] > 3,"Medium Value Customer",
                            np.where(customer_rfm['RFM_Score'] > 1.6,'Low Value Customers', 'Lost Customers'
                                     ))))))
    #Grafik akhir analisis RFM
    explode=(0.1, 0.1, 0.1, 0.1)
    fig, ax =plt.subplots()
    ax.pie(
        customer_rfm.Customer_segment.value_counts(),
		labels=customer_rfm.Customer_segment.value_counts().index,
		autopct='%.0f%%',
		explode=explode)
    st.pyplot(fig)

#streamlit run C:\Users\LENOVO\Downloads\hello-world\submission\dashboard\dashboard.py


st.caption('ZakkiZainalAbidinFadhilurrahman 2023')

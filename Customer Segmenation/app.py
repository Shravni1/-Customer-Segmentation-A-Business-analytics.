import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
import pickle
import streamlit as st
import os
from datetime import datetime
import squarify
import base64
from io import BytesIO
import altair as alt
import numpy as np
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules    
import matplotlib.backends.backend_pdf as pdf_backend


# Initializing session state variables
st.session_state['df'] = st.session_state.get('df', None)
st.session_state['uploaded_file'] = st.session_state.get('uploaded_file', None)


# Function to download the visualizations as a PDF
def download_visualizations_as_pdf(visualizations):
    pdf_bytes = BytesIO()

    with pdf_bytes as pdf:
        plt.figure(figsize=(10, 6))
        for visualization in visualizations:
            visualization()  # Call the visualization function
            plt.savefig(pdf, format='pdf')
            plt.clf()  # Clear the figure for the next visualization

    pdf_bytes.seek(0)
    b64_pdf = base64.b64encode(pdf_bytes.read()).decode()

    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="visualizations.pdf">Download Visualizations as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)


def visualize_predicted_clusters(df_RFM):
    st.title("Predicted Clusters for New Data")
    
    # Create a bar chart to show the distribution of predicted clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_RFM, x='Cluster', palette='viridis')
    plt.title('Distribution of Predicted Clusters for New Data', fontsize=16)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Display the plot in Streamlit
    st.pyplot(plt)
if 'df_new' not in st.session_state:
    st.session_state['df_new'] = pd.DataFrame(columns=['InvoiceNo','StockCode','ProductsCategory','Quantity','InvoiceDate','UnitPrice','CustomerID','Cities','Tax 5%','Total','Gender','Payment'])


# Function to download the visualizations as a PDF
def download_as_pdf(visualizations):
    pdf_bytes = BytesIO()

    with pdf_backend.PdfPages(pdf_bytes) as pdf:
        for visualization in visualizations:
            visualization()  # Call the visualization function
            plt.savefig(pdf, format='pdf', bbox_inches='tight')
            plt.clf()  # Clear the figure for the next visualization

    pdf_bytes.seek(0)
    b64_pdf = base64.b64encode(pdf_bytes.read()).decode()

    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="visualizations.pdf">Download Visualizations as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
    










# GUI setup
st.title("📊 Customer Insights 📈")
st.header("Unlocking Customer Segmentation", divider='rainbow')
menu = ["About Us", "Upload Data","Data Understanding","Data Visualization","Modeling & Predictions","Summary","Feedback"] # , "BigData: Spark"
choice = st.sidebar.selectbox('Menu', menu)

# Function to load data from CSV file
def load_data(uploaded_file):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        df = pd.read_csv(uploaded_file)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])  # Convert to datetime
        st.session_state['df'] = df
        return df
    else:
        st.write("Please upload a CSV file to proceed.")
        return None

# Function to create a download link for CSV
def csv_download_link(df, csv_file_name, download_link_text):
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{csv_file_name}">{download_link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)    

# Initializing session state variables
if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None


    


if choice == 'About Us':
   
    st.write("""
    Welcome to Customer Insights, your trusted partner in business analytics and data-driven solutions.
    We are dedicated to helping small-scale businesses thrive in today's competitive market.""")
    st.image("target.jpg", caption="Customer Segmentation", use_column_width=True)
    st.write("""At Customer Insights, we specialize in three core services:

    1. Market Basket Analysis: We provide valuable insights into customer purchase patterns, helping you understand which products are often bought together. This information can be a game-changer in your inventory management and marketing strategies.

    2. Customer Segmentation: Our advanced algorithms segment your customer base, allowing you to tailor your marketing efforts to different groups with unique preferences and behaviors. This leads to more effective and targeted marketing campaigns.

    3. Day-to-Day Transaction Tracking: We offer a user-friendly platform for tracking your daily business transactions. You can effortlessly record sales, expenses, and revenue, making financial management a breeze.

    With a team of experienced data analysts and a passion for helping small businesses succeed, we are committed to providing you with the tools and insights needed to make informed decisions and grow your business.

    Experience the power of data: Our interactive tools and dashboards empower you to explore your data and gain valuable insights with ease.
    """)
    st.write("Contact Us")

# Create three columns layout
    left_column, middle_column, right_column = st.columns(3)

# Left column - Email
    left_column.subheader("📧 Email")
    left_column.markdown("[customerinsights@.com](mailto:customerinsights@.com)")

# Middle column - Phone
    middle_column.subheader("☎️ Phone")
    middle_column.markdown("[+91 7207345261](tel:+917207345261)")

# Right column - Instagram
    right_column.subheader("📷 Instagram")
    right_column.markdown("[Follow us on Instagram](https://www.instagram.com/customerinsights/)")

    


elif choice == 'Upload Data':    
    # Allow user to upload a CSV file
    st.sidebar.write("### Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

    # If file is uploaded, load the data and display a preview
    if uploaded_file is not None:
        
        st.session_state['uploaded_file'] = uploaded_file
        load_data(uploaded_file)

    if st.session_state['df'] is not None:
        st.write("### Data Overview")
        st.write("Number of rows:", st.session_state['df'].shape[0])
        st.write("Number of columns:", st.session_state['df'].shape[1])
        st.write("First five rows of the data:")
        st.write(st.session_state['df'].head())

        
        

        # Add more analysis options based on user needs

        # Create a download link for the processed data
        st.sidebar.write("### Download Processed Data")
        csv_download_link(st.session_state['df'], "processed_data.csv", "Download Processed Data")
    # ...

# ...

# ...
elif choice == 'Data Understanding':
    st.write("### Data Cleaning")
    
    if st.session_state['df'] is not None:
        # 1. Handling missing, null, and duplicate values
        st.write("Number of missing values:")
        st.write(st.session_state['df'].isnull().sum())

        st.write("Number of NA values:")
        st.write((st.session_state['df'] == 'NA').sum())

        st.write("Number of duplicate rows:", st.session_state['df'].duplicated().sum())

        # Providing options for handling missing and duplicate values
        if st.checkbox('Remove duplicate rows'):
            st.session_state['df'].drop_duplicates(inplace=True)
            st.write("Duplicate rows removed.")
        
        if st.checkbox('Remove rows with NA values'):
            st.session_state['df'].replace('NA', pd.NA, inplace=True)
            st.session_state['df'].dropna(inplace=True)
            st.write("Rows with NA values removed.")

        # 2. Display number of unique values for each column
        st.write("Number of unique values for each column:")
        st.write(st.session_state['df'].nunique())


elif choice == 'Data Visualization':
    st.header("Data  Visualizations📊 ")


    # Access the DataFrame selected in the "Data Understanding" section
    df = st.session_state['df']
    
    
    

# Visualization
    st.subheader("Top 5 Frequently Purchased Products")

# Count occurrences of each product category
    top_products = df['ProductsCategory'].value_counts().iloc[:5]

# Plot the top 5 frequently purchased products using Streamlit
    st.bar_chart(top_products)


    # Visualize Number of Transactions Occurred Each Day of the Week
    st.title("Number of Transactions Occurred Each Day of the Week")
    # Convert 'InvoiceDate' column to datetime if it's not already
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Extract the day of the week from 'InvoiceDate' (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek

    # Group by day of the week and count unique 'InvoiceNo' values
    order_day = df.groupby('DayOfWeek')['InvoiceNo'].nunique()

    # Create a bar plot using Matplotlib
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=order_day.index, y=order_day.values, palette="Set3")
    ax.set_title('Number of Transactions Occurred Each Day', size=20)
    ax.set_xlabel('Day', size=14)
    ax.set_ylabel('Number of Transactions', size=14)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Display the Matplotlib plot using Streamlit
    st.pyplot(plt)

    # Visualize Number of Transactions Occurred Each Hour
    st.title("Number of Transactions Occurred Each Hour")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Calculate the number of transactions per hour
    order_hour = df.groupby(df['InvoiceDate'].dt.hour)['InvoiceNo'].nunique().reset_index()

    # Create a bar plot using Matplotlib
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='InvoiceDate', y='InvoiceNo', data=order_hour, palette="colorblind")
    ax.set_title('Number of Transactions Occurred Each Hour', size=20)
    ax.set_xlabel('Hour', size=14)
    ax.set_ylabel('Number of Transactions', size=14)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)

    # Display the Matplotlib plot using Streamlit
    st.pyplot(plt)
    


    # Visualize Number of Transactions Occurred Each Day
    st.title("Number of Transactions Occurred Each Day")
    # Convert 'InvoiceDate' column to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Calculate the number of transactions per day
    order_day = df.groupby(df['InvoiceDate'].dt.date)['InvoiceNo'].nunique()

    # Create a bar plot for day-to-day transactions
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=order_day.index, y=order_day.values, palette="colorblind")
    ax.set_title('Number of Transactions Occurred Each Day', size=20)
    ax.set_xlabel('Date', size=14)
    ax.set_ylabel('Number of Transactions', size=14)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    plt.xticks(rotation=45)

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Visualize Top 5 Customers by Total Spending
    st.title("Top 5 Customers by Total Spending")
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Display the updated DataFrame
    st.write("DataFrame with TotalPrice Column:")
    st.write(df.head())

    # Calculate the total spending per customer using a pivot table
    price_cust = pd.pivot_table(df, index='CustomerID', values='TotalPrice', aggfunc=np.sum)
    # Sort the customers by total spending in descending order and get the top 5
    top_customers = price_cust.nlargest(5, 'TotalPrice')
    # Create a horizontal bar chart using Altair
    chart = alt.Chart(top_customers.reset_index()).mark_bar().encode(
        y=alt.Y('CustomerID:N', title='Customer ID'),
        x=alt.X('TotalPrice:Q', title='Total Spending', scale=alt.Scale(domain=(0, max(top_customers['TotalPrice']) * 1.1))),
        color=alt.Color('CustomerID:N', legend=None),
        tooltip=['CustomerID', 'TotalPrice']
    ).properties(
        width=600,
        height=400
    )

    # Display the Altair chart using Streamlit
    st.write("The following are the top 5 customers who spend the most money on Online Retail:")
    st.altair_chart(chart, use_container_width=True)


    # Visualize Top 5 Products Categories vs Gender
    st.title("Top 5 Products Categories vs Gender")
    # Calculate the top 5 product categories by count
    top_categories = df['ProductsCategory'].value_counts().head(5).index

    # Filter the DataFrame for the top 5 categories
    df_filtered = df[df['ProductsCategory'].isin(top_categories)]

    # Create a count plot
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_filtered, x='ProductsCategory', hue='Gender', palette='Set1')
    plt.title('Top 5 Products Categories vs Gender', fontsize=16)
    plt.xlabel('ProductsCategory', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Gender', fontsize=12, title_fontsize=12)

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Visualize Cities vs Top 5 Products Categories
    st.title("Cities vs Top 5 Products Categories")
     # Get the top 5 product categories
    top_categories = df['ProductsCategory'].value_counts().head(5).index

    # Filter the DataFrame for the top 5 categories
    df_filtered = df[df['ProductsCategory'].isin(top_categories)]

    # Filter data for the selected city
    selected_city = st.selectbox("Select a City:", df['Cities'].unique())
    filtered_df = df_filtered[df_filtered['Cities'] == selected_city]

    # Create a count plot to visualize selected city vs top 5 product categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='Cities', hue='ProductsCategory', palette='Set1')
    plt.title(f'{selected_city} vs Top 5 Products Categories', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='ProductsCategory', fontsize=12, title_fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(plt)
    # Visualize Payment Modes
    st.title("Payment Modes")
    # Group data by PaymentMode and calculate total amounts
    payment_mode_totals = df.groupby('Payment')['Total'].sum()

    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(payment_mode_totals, labels=payment_mode_totals.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

    # Display the pie chart in Streamlit
    
    st.pyplot(fig)
    
    
    
   
  
    
 
    
    
    
    
    
    
elif choice == 'Modeling & Predictions':
    st.header("🤖 Modeling and Predictions 📈")


   # Access the DataFrame selected in the "Data Understanding" section
    df = st.session_state['df']

    
    st.title("Customer Loyalty")
    # Define the analysis date
    analysis_date = pd.to_datetime('2022-01-01')

    # Check if data is available
    if df is not None:
        # Check if 'TotalPrice' column exists, if not, calculate it
        if 'TotalPrice' not in df.columns:
            # Assuming TotalPrice is calculated based on other columns (adjust as needed)
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

        # Calculate RFM metrics for customers
        rfm_cust = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (analysis_date - pd.to_datetime(x.max())).days,
            'InvoiceNo': 'count',
            'TotalPrice': 'sum'
        })

        # Rename columns
        rfm_cust.rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'TotalPrice': 'Monetary'
        }, inplace=True)

        # Calculate loyalty level based on quartiles
        loyalty_quartile = pd.qcut(rfm_cust['Recency'], q=4, labels=['Platinum', 'Gold', 'Silver', 'Bronze'])
        rfm_cust['Loyalty_Level'] = loyalty_quartile.values

        # Display the RFM metrics DataFrame
        st.write("RFM Metrics for Customers:")
        st.write(rfm_cust)

        # Display a bar plot of Loyalty Levels
        loyalty_counts = rfm_cust['Loyalty_Level'].value_counts()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=loyalty_counts.index, y=loyalty_counts.values, palette='Set1')
        plt.title('Customer Loyalty Levels', fontsize=16)
        plt.xlabel('Loyalty Level', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        st.pyplot(plt)

    else:
        st.write("Please upload a CSV file with the necessary columns and perform data understanding before data preparation.")
        
       

    # Include the rest of your Modeling & Evaluation code here
    
    st.title("Customer Segmentation")

    # Calculate RFM metrics for customers
    rfm_cust = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (analysis_date - pd.to_datetime(x.max())).days,
        'InvoiceNo': 'count',
        'Total': 'sum'  # Replace 'TotalPrice' with 'Total'
    })

    # Rename columns
    rfm_cust.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'Total': 'Monetary'  # Replace 'TotalPrice' with 'Total'
    }, inplace=True)

    # Example: Define and fit a K-Means model
   
    num_clusters = 4  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    rfm_cust['Segment'] = kmeans.fit_predict(rfm_cust[['Recency', 'Frequency', 'Monetary']])

    # Display the RFM metrics DataFrame with segments
    st.write("RFM Metrics for Customers with Segments:")
    st.write(rfm_cust)

    # Include your third visualization here
    st.title("3D Customer Segmentation Visualization")

    # Import necessary libraries for 3D plotting
    import plotly.express as px
    import plotly.graph_objects as go

    # Create a 3D scatter plot
    fig = px.scatter_3d(
        rfm_cust,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Segment',  # Use 'Segment' for coloring
        symbol='Segment',  # Use 'Segment' for symbols
        labels={'Segment': 'Customer Segment'},
        title='Customer Segmentation in 3D',
    )

    # Customize the layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Recency',
            yaxis_title='Frequency',
            zaxis_title='Monetary'
        )
    )

    # Display the interactive 3D plot
    st.plotly_chart(fig)
    
    



 # Market Basket Analysis
    st.title("Market Basket Analysis - Frequent Itemsets and Association Rules")

    # Split the ProductsCategory into a list
    df['ProductsCategory'] = df['ProductsCategory'].str.split(', ')

    # Use one-hot encoding to create the basket_sets dataset
    basket_sets = pd.get_dummies(df['ProductsCategory'].apply(pd.Series).stack(), prefix='', prefix_sep='').max(level=0)

    # Sidebar for user input
    st.sidebar.title("Market Basket Analysis Parameters")
    
    
    min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.02, step=0.01)


    # Find frequent itemsets using Apriori
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

    # Display frequent itemsets
    st.title("Frequent Itemsets:")
    st.write(frequent_itemsets)

  
   
    




elif choice == 'Summary':
    st.header("Sales Analysis - Executive Summary")
    st.write("Here is Your Sales Analysis", divider='star')

    if st.session_state['df'] is not None:
        df_selection = st.session_state['df']

        # Code for data understanding section
        # ...

        total_sales = int(df_selection["Total"].sum())
        total_costs = int(df_selection["Total"].sum())  # Placeholder, replace with actual cost calculation
        total_profit = total_sales - total_costs

        average_sale_by_transaction = round(df_selection["Total"].mean(), 2)
        daily_customers = df_selection.groupby(df_selection['InvoiceDate'].dt.date)['CustomerID'].nunique().reset_index()
        left_column, right_column, bottom_column = st.columns(3)

        with left_column:
            st.subheader("Total Sales:")
            st.subheader(f"Cr {total_sales:,}")

        with right_column:
            st.subheader("Daily Number of Customers:")
            st.subheader(f"{daily_customers['CustomerID'].sum():,}")

        with bottom_column:
            st.subheader("Average Sales Per Transaction:")
            st.subheader(f" {average_sale_by_transaction}")

        # Code for data upload or modification
        # ...

        # SALES BY PRODUCT LINE [BAR CHART]
        if 'ProductsCategory' in df_selection.columns:
            # SALES BY PRODUCT CATEGORY [BAR CHART]
            sales_by_category = df_selection.groupby(by=["ProductsCategory"])[["Total"]].sum().sort_values(
                by="Total", ascending=False).head(10)
            fig_category_sales = px.bar(
                sales_by_category,
                x="Total",
                y=sales_by_category.index,
                orientation="h",
                title="<b>Top 10 Products by Sales</b>",
                color_discrete_sequence=["#0083B8"] * len(sales_by_category),
                template="plotly_white",
            )
            fig_category_sales.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=(dict(showgrid=False))
            )

            # SALES BY CITIES [TREEMAP]
            sales_by_city = df_selection.groupby(by=["Cities"])[["Total"]].sum().sort_values(
                by="Total", ascending=False).head(5)
            fig_city_sales_treemap = px.treemap(
                sales_by_city.reset_index(),  # Reset the index to include 'Cities' as a column
                path=["Cities"],  # Specify the hierarchical path
                values="Total",
                title="<b>Top 5 Cities by Sales</b>",
                color_discrete_sequence=["#0083B8"] * len(sales_by_city),
                template="plotly_white",
            )
            fig_city_sales_treemap.update_layout(
                plot_bgcolor="rgba(200, 200, 200, 0.2)",  # Light gray background
            )

            # Streamlit app layout
            st.title("Sales Analysis")
            left_column, right_column = st.columns(2)

            # Display charts in Streamlit app
            left_column.plotly_chart(fig_category_sales, use_container_width=True)
            right_column.plotly_chart(fig_city_sales_treemap, use_container_width=True)

            # Distribution of Top 5 Product Categories vs Gender
            gender_category_distribution = pd.crosstab(df_selection['ProductsCategory'], df_selection['Gender'])
            fig_gender_category_distribution = px.bar(
                gender_category_distribution,
                barmode='group',
                title='<b>Top 5 Product Categories vs Gender Distribution</b>',
                color_discrete_sequence=['#0083B8', '#00B4D8'],  # Blue for Male, Light Blue for Female
                template='plotly_white'
            )

            # Track Weekly Transactions
            weekly_transactions = df_selection.resample('W-Mon', on='InvoiceDate').agg(
                {'InvoiceNo': 'count'}).reset_index()
            fig_weekly_transactions = px.line(
                weekly_transactions,
                x='InvoiceDate',
                y='InvoiceNo',
                title='<b>Weekly Transactions</b>',
                labels={'InvoiceNo': 'Number of Transactions'},
                template='plotly_white',
                line_shape='linear',  # Use linear line shape for a smooth line
                markers=True,
                color_discrete_sequence=['#87CEEB']  # Sky Blue color
            )

            # Display charts in Streamlit app
            left_column.plotly_chart(fig_gender_category_distribution, use_container_width=True)
            right_column.plotly_chart(fig_weekly_transactions, use_container_width=True)

            if 'CustomerID' in df_selection.columns:
                # Assuming 'Total' and 'Number of Purchases' are relevant metrics for loyalty
                customer_loyalty = df_selection.groupby('CustomerID').agg({
                    'Total': 'sum',
                    'InvoiceNo': 'nunique'  # Number of unique purchases
                })

                # Assuming 'Average Transaction Value' is another loyalty metric
                customer_loyalty['AvgTransactionValue'] = customer_loyalty['Total'] / customer_loyalty['InvoiceNo']

                # Define Loyalty Criteria
                high_loyalty_threshold = 10000  # You can adjust this threshold based on your business needs
                low_loyalty_threshold = 5000  # You can adjust this threshold based on your business needs

                # Identify Loyal Customers
                loyal_customers = customer_loyalty[customer_loyalty['Total'] > high_loyalty_threshold]

                # Identify Customers Needing Attention
                attention_customers = customer_loyalty[
                    (customer_loyalty['Total'] <= high_loyalty_threshold) & (customer_loyalty['Total'] > low_loyalty_threshold)
                ]

                # Identify Customers for Promotions
                promo_customers = customer_loyalty[customer_loyalty['AvgTransactionValue'] < 50]  # You can adjust this threshold based on your business needs

                # Identify Best Customers
                best_customers = customer_loyalty.nlargest(5, 'Total')  # Display top 5 customers by total purchase amount

                # Identify Promising Customers for Better Deals
                promising_customers = customer_loyalty[
                    (customer_loyalty['Total'] <= high_loyalty_threshold) & (customer_loyalty['Total'] > low_loyalty_threshold) &
                    (customer_loyalty['AvgTransactionValue'] > 50)
                ]
                # Combine all categories into a single DataFrame
                all_customers = pd.concat([
                    loyal_customers.assign(Category='Loyal Customers'),
                    attention_customers.assign(Category='Customers Needing Attention'),
                    promo_customers.assign(Category='Customers for Promotions'),
                    best_customers.assign(Category='Best Customers'),
                    promising_customers.assign(Category='Promising Customers')
                ])
                # Create a pie chart
                fig_pie = px.pie(all_customers, names='Category', title='Customer Loyalty Distribution')

                # Display the pie chart
                st.plotly_chart(fig_pie, use_container_width=True)
                
                
elif choice == "Feedback":
    # User Feedback section
    st.write("### User Feedback")
    user_feedback = st.text_area("Please share your comments or feedback:", value='')

    if st.button("Submit Feedback"):
        # Store the feedback with timestamp in a DataFrame
        current_time = datetime.now()
        feedback_df = pd.DataFrame({
            'Time': [current_time],
            'Feedback': [user_feedback]
        })

        # Check if feedback file already exists
        if not os.path.isfile('feedback.csv'):
            feedback_df.to_csv('feedback.csv', index=False)
        else: # Append the new feedback without writing headers
            feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)

        st.success("Your feedback has been recorded! Thanks for choosing us. We hope you have a nice day!")

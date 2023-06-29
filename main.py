import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import logging
import base64
import plotly.express as px
import plotly.graph_objects as go
import xlsxwriter
import io
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

logging.basicConfig(level=logging.ERROR)

# Function to check if the username and password are valid
def authenticate(username, password):
    response = requests.post("http://localhost:1000/authenticate", json={"username": username, "password": password})

    return response.status_code == 200 and response.json().get("authenticated", False)


# Page function for the authentication page
def authentication_page():
    if "auth" not in st.session_state or not st.session_state["auth"]:
        st.write("Please sign in")
        st.session_state["page"] = None
        # Collect username and password from the user
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")



        # Check if the username and password are valid
        if st.button("Login"):
            if authenticate(username, password):
                # Clear the session state cache
                st.session_state.clear()
                # Store authentication status in session_state
                st.session_state["authenticated"] = True
                st.session_state["page"] = "second"
                st.experimental_rerun()  # Rerun the app to display the second page
            else:
                st.error("Invalid username or password")

# Page function for the second page
def second_page():
    st.write("Please upload a file and select appropriate parameters for Machine Learning Forecasting")
    
        # Bookmark button
    if st.button("Bookmark"):
        st.session_state.file_data = pd.read_excel("ics.xlsx")
        st.session_state.date_column = "Date"
        st.session_state.regressor_columns = ['Covid', 'Price', 'other']
        st.session_state.products_column = "Sku"
        st.session_state.forecast_column = "Units"
        st.session_state.predicting_year = 2023
        st.session_state.predicting_month = 1
        st.session_state.user_prediction = "AOP"
        st.session_state.products_to_forecast = "Products"
        # products_list = pd.read_excel("ics.xlsx", sheet_name='Forecast_sku')
        # st.session_state.products_to_forecast = products_list['Sku'].unique()


    # File upload
    st.write("Upload a file")
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])

    if "file_data" not in st.session_state:
        st.session_state.file_data = None

    if uploaded_file is not None and st.session_state.file_data is None:
        # Read the uploaded file according to its type
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            st.session_state.file_data = pd.read_excel(uploaded_file)
        elif uploaded_file.type == "text/csv":
            st.session_state.file_data = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type: " + uploaded_file.type)
            return

        st.write("File content:")
        st.write(st.session_state.file_data)

    if st.session_state.file_data is not None:
        
        # Column selection
        columns = st.session_state.file_data.columns.tolist()
        date_column = st.selectbox("Select date column", columns, index=columns.index(st.session_state.date_column) if "date_column" in st.session_state else 0)
        regressor_columns = st.multiselect("Select regressors/features for forecasting", columns, default=st.session_state.regressor_columns if "regressor_columns" in st.session_state else None)
        products_column = st.selectbox("Select a product column", columns, index=columns.index(st.session_state.products_column) if "products_column" in st.session_state else 0)
        forecast_column = st.selectbox("What you want to forecast", columns, index=columns.index(st.session_state.forecast_column) if "forecast_column" in st.session_state else 0)
        predicting_year = st.text_input("Enter the year to forecast", value=str(st.session_state.predicting_year) if "predicting_year" in st.session_state else '')
        try:
            predicting_year = int(predicting_year) if predicting_year.isdigit() else None
        except:
            pass
        predicting_month = st.text_input("Enter the month to forecast", value=str(st.session_state.predicting_month) if "predicting_month" in st.session_state else '')
        try:
            predicting_month = int(predicting_month) if predicting_month.isdigit() else None
        except:
            pass
        user_prediction = st.selectbox("Select your forecasted column", columns, index=columns.index(st.session_state.user_prediction) if "user_prediction" in st.session_state else 0)
        products_to_forecast = st.selectbox("Select products column to forecast", columns, index=columns.index(st.session_state.products_to_forecast) if "products_to_forecast" in st.session_state else 0)


        st.write("Selected date column:", date_column)
        st.write("Selected regressors/features:", regressor_columns)
        st.write("Selected products column", products_column)
        st.write("Selected forecast column:", forecast_column)
        st.write("Selected year to forecast", predicting_year)
        st.write("Selected month column to predict", predicting_month)
        st.write("Selected your forecasted column", user_prediction)
        st.write("Selected your products column to forecast", products_to_forecast)

        

        # Convert timestamp values to string representations
        if st.session_state.file_data[date_column].dtype == pd.Timestamp:
            st.session_state.file_data[date_column] = st.session_state.file_data[date_column].astype(str)

        # Prepare the data to send to the backend
        data = {
            "file_content": st.session_state.file_data.to_json(date_format='iso', orient='records').replace("\n", "\\n"),
            "date_column": str(date_column),  # Convert date_column to string
            "regressor_columns": regressor_columns,  # Convert regressor_columns to list of strings
            "products_column": str(products_column),  # Convert products_column to string
            "forecast_column": str(forecast_column),  # Convert forecast_column to string
            "forecast_year": int(predicting_year) if predicting_year is not None else None,  # Convert predicting year to int
            "forecast_month": int(predicting_month) if predicting_month is not None else None,  # Convert predicting year to int
            "user_prediction": str(user_prediction),
            "products_to_forecast": str(products_to_forecast)
        }

                # Button to preprocess data and create Algorithms
        if st.button("Preprocess data and create Algorithms to Forecasting"):
            st.session_state.date_column = date_column
            st.session_state.regressor_columns = regressor_columns
            st.session_state.products_column = products_column
            st.session_state.forecast_column = forecast_column
            st.session_state.user_prediction = user_prediction
            st.session_state.predicting_year = predicting_year
            st.session_state.predicting_month = predicting_month
            st.session_state.products_to_forecast = products_to_forecast
            st.session_state["proceed"] = False
            # Create an empty element
            status_placeholder = st.empty()

            # Show emoji while waiting for the response
            status_placeholder.text("Processing data, wait please ⌛️")

            # Send the data to the backend for data processing
            response = requests.post("http://localhost:1000/output", json=data)

            if response.status_code == 200:
                processed_data = response.json()  # Get the processed data from the response
                st.session_state["processed_data"] = processed_data
                # Clear the status placeholder
                status_placeholder.empty()
                st.success("Great, everything is ready!")  # Success message
                st.session_state["page"] = "third"
                st.experimental_rerun() 
            else:
                # st.error(
                #     "Error processing data. Please try again. Error message: {}".format(response.text)
                # )
                st.session_state["page"] = "second"
            # Clear the status placeholder
            status_placeholder.empty()

# Function to export data to Excel and create a download link
def export_to_excel(data):
    # Create a BytesIO buffer
    excel_buffer = io.BytesIO()

    # Write the data to the BytesIO buffer as an Excel file
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as excel_writer:
        data.to_excel(excel_writer, sheet_name="Forecast Data", index=False)

    # Seek to the beginning of the buffer
    excel_buffer.seek(0)

    # Create a download link for the Excel file
    b64_excel = base64.b64encode(excel_buffer.read()).decode("utf-8")
    href = f'<a href="data:application/octet-stream;base64,{b64_excel}" download="forecast_data.xlsx">Download Excel file</a>'

    return href



def third_page():
    if "processed_data" in st.session_state:
        data = pd.read_json(st.session_state["processed_data"])
        data['ds'] = pd.to_datetime(data['ds']).dt.strftime('%Y-%m-%d')
        columns = ['ds'] + [st.session_state.products_column] + [x for x in st.session_state.regressor_columns] + ['y'] + [st.session_state.user_prediction] + ['BestModel', 'Predicted_BestModel', 'Accuracy_BestModel'] + [f'{st.session_state.user_prediction} accuracy']

        try:
            # Calculate average accuracy and average AOP accuracy
            average_accuracy = data['Accuracy_BestModel'].str.rstrip('%').astype(float).mean()
            average_user_accuracy = data[f'{st.session_state.user_prediction} accuracy'].str.rstrip('%').astype(float).mean()
            st.header("Prediction Results")
            # Display average accuracy and average AOP accuracy
            st.write(f"Average Accuracy: {average_accuracy:.1f}%")
            st.write(f"Average AOP Accuracy: {average_user_accuracy:.1f}%")
        except Exception as e:
            pass
        data_to_show = data[columns]  
        # Rename columns
        data_to_show = data_to_show.rename(columns={'ds': 'Date', 'y': 'Fact'}) 
        st.write(data_to_show)
        # Add export button
        if st.button("Export to Excel"):
            excel_href = export_to_excel(data)
            st.markdown(excel_href, unsafe_allow_html=True)

        try:
            for product in data[st.session_state.products_column].unique():
                product_data = data[data[st.session_state.products_column] == product]


                # Calculate average accuracy and average accuracy of the user
                average_accuracy = product_data['Accuracy_BestModel'].str.rstrip('%').astype(float).mean()
                average_user_accuracy = product_data[f'{st.session_state.user_prediction} accuracy'].str.rstrip('%').astype(float).mean()
                best_model = product_data['BestModel'].iloc[0]

                # Display average accuracy and average accuracy of the user
                # st.subheader(f"Product: {product}")
                st.write(f"{product} accuracy")
                st.write(f"Average Accuracy: {average_accuracy:.1f}%")
                st.write(f"Average User Accuracy: {average_user_accuracy:.1f}%")
                st.write(f"Best Model: {best_model}")




                # Create the figure
                fig = px.line(product_data, x='ds', y='y', title=product)

                # Add Forecasted line
                fig.add_scatter(x=product_data['ds'], y=product_data['Predicted_BestModel'], mode='lines', name='BestModel', line=dict(color='green'))

                # Add AOP line
                fig.add_scatter(x=product_data['ds'], y=product_data[st.session_state.user_prediction], mode='lines', name='UserForecast', line=dict(color='blue'))

                # Add line for 'y'
                fig.add_scatter(x=product_data['ds'], y=product_data['y'], mode='lines', name='Fact', line=dict(color='grey'))

                # Update layout and settings
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title='Sales in thousands',
                    title_x=0.5,  # Align title in the middle
                    title_y=0.95,  # Set the y position of the title
                    yaxis_tickformat=".3s"  # Set tick format to thousands (e.g., 2.3k)
                )

                # Highlight specific values
                highlighted_values = [product_data['y'].max()]
                fig.add_trace(
                    px.scatter(product_data, x='ds', y='y')
                        .update_traces(marker=dict(color='red'), selector=dict(mode='markers'))
                        .data[0]
                )
                fig.update_traces(selectedpoints=highlighted_values, selector=dict(type='scatter'), marker=dict(color='red'))

                # Adjust y-axis range to fit data points
                y_axis_range = [product_data['y'].min(), product_data['y'].max()]
                y_axis_padding = 0.1 * (y_axis_range[1] - y_axis_range[0])  # Add padding to the range
                fig.update_layout(yaxis=dict(range=[y_axis_range[0] - y_axis_padding, y_axis_range[1] + y_axis_padding]))
                # Fit the graph to the data
                fig.update_layout(autosize=True)
                st.plotly_chart(fig)

                # Create the correlation matrix
                columns_to_correlate = st.session_state.regressor_columns + [f"Month_{i}" for i in range(1, 13)] + ['y']
                correlation_data = product_data[columns_to_correlate].corr()

                # Filter correlation coefficients for 'y' column only
                y_correlations = correlation_data['y'].dropna()

                # Remove 'y' variable
                y_correlations = y_correlations.drop('y')

                # Calculate absolute values of correlation coefficients
                y_correlations_abs = y_correlations.abs()

                # Sort by absolute values using mod() function
                sorted_indices = np.argsort(np.mod(y_correlations_abs, 1))

                # Create a new DataFrame with sorted correlation coefficients
                corr_df = pd.DataFrame({'Variable': y_correlations.index[sorted_indices],
                                        'Correlation Coefficient': y_correlations.values[sorted_indices]})

                # Set smaller table display options
                               # Set smaller table display options
                table_html = corr_df.to_html(index=False, classes=["small-table", "striped"])

                # Display the table
                st.markdown(
                    f"""
                    <div class="table-container">
                        {table_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <style>
                    .table-container {{
                        margin-left: 0;
                    }}
                    .small-table {{
                        font-size: 12px;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:
            pass
    else:
        st.write("Please upload data and preprocess!")






# Main app logic
def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "first"


    # # Read the background image file
    # with open('image.jpg', 'rb') as image_file:
    #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # # Add CSS to set the background image
    # st.markdown(
    #     f"""
    #     <style>
    #     .stApp {{
    #         background-image: url(data:image/jpeg;base64,{encoded_string});
    #         background-size: cover;
    #         background-repeat: no-repeat;
    #         background-position: center;
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
    st.markdown(
                """
                <h1 style="text-align: center; font-size: 32px; font-weight: bold; padding-top: 20px;">
                Welcome to Teva's Analytical website
                </h1>
                """,
                unsafe_allow_html=True
                )

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

        # Sidebar buttons
    if st.sidebar.button("Authentication"):
        try:
            del st.session_state["auth"]
        except:
            pass
        st.session_state["page"] = "authentication_page"
        


    if st.sidebar.button("Second Page"):
        if not st.session_state["page"]:
            st.session_state["page"] = "authentication_page"
        else:
            st.session_state["page"] = "second"

    if st.sidebar.button("Third Page"):
        if not st.session_state["page"]:
            st.session_state["page"] = "authentication_page"
        else:
            st.session_state["page"] = "third" 

    # Page 1: Authentication
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        authentication_page()

    # Page 2: Second page
    if st.session_state["page"] == "second":
        second_page()
    if st.session_state["page"] == "third":
        third_page()

# Run the app
if __name__ == "__main__":
    main()

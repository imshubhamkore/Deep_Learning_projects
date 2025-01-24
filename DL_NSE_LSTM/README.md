## National Stock Exchange 

## Overview
This project utilizes an LSTM (Long Short-Term Memory) neural network model to predict the future stock prices based on historical stock data. It uses various preprocessing techniques, feature engineering, and deep learning to make stock price predictions.

## Steps:

1. **Data Preprocessing**:
   - **Dataset Loading**: Read the dataset and filter it based on the stock symbol (`3IINFOTECH`) and series (`EQ`).
   - **Datetime Parsing**: Convert the `TIMESTAMP` column to datetime and set it as the index.
   - **Drop Unnecessary Columns**: Drop columns like `SYMBOL`, `SERIES`, and `ISIN` which are not needed for the analysis.

2. **Feature Selection**:
   - **Correlation Analysis**: Use a correlation heatmap to visualize the relationships between numerical features like `OPEN`, `CLOSE`, `HIGH`, `LOW`, `PRICE_CHANGE`, `VOLATILITY`, etc.
   - **Feature Selection**: Select relevant features for model input.

3. **Data Scaling**:
   - **MinMaxScaler**: Apply MinMax scaling to scale the features between 0 and 1. This is crucial for LSTM models to handle numerical data properly.
   - **Train-Test Split**: Split the data into training and testing sets, using the scaled data.

4. **Sequence Creation**:
   - **Window-based Input**: Create time-sequenced data (`X_train`, `y_train`, `X_test`, `y_test`) based on a `window_size` (60 in this case). This helps the LSTM model to capture temporal dependencies.

5. **Model Construction**:
   - **LSTM Layers**: Construct an LSTM model with two Bidirectional LSTM layers. The first LSTM layer is `return_sequences=True` to output sequences to the next layer. The second LSTM layer.
   - **Dropout Layer**: Add a `Dropout` layer with a rate of 0.3 to reduce overfitting.
   - **Dense Layer**: The output layer is a `Dense` layer with 1 unit to predict a single value (price).

6. **Model Compilation**:
   - **Optimizer**: Compile the model using the `Adam` optimizer.
   - **Loss Function**: Use `mean_squared_error` as the loss function, which is typical for regression tasks.

7. **Model Training**
   - **Epochs & Batch Size**: Train the model for a defined number of epochs (50) and use a batch size of 32.
   - **Validation Data**: Validate the model on the test data during training to monitor its performance.

8. **Model Evaluation**:
   - **Predictions**: Predict stock prices on both the training and testing data.
   - **Inverse Scaling**: Use the `inverse_transform` method of the scaler to convert the predicted values back to the original scale.
   - **Error Metrics**: Compute evaluation metrics like `RMSE` (Root Mean Squared Error), `MAE` (Mean Absolute Error), and `R² score` for both training and test data.

9. **Future Price Prediction**:
   - **Sliding Window**: Use the last `window_size` days of data as input to predict the future stock prices for the next 30 days.
   - **Future Predictions**: For each day, the model predicts the next price, and the input is updated with the predicted price for the next iteration.
   - **Generate Future Dates**: Generate future dates based on the last date in the test data and combine the past and predicted prices.

10. **Visualization**:
    - **Combining Past and Future Data**: Combine the actual past prices with the predicted future prices.
    - **Plotting**: Optionally visualize the combined data on a plot to see the trend of predicted prices.


## Conclusion:
- The LSTM model is used to predict future stock prices based on historical data.
- The model's performance is evaluated using metrics such as RMSE, MAE, and R² score.
- The future prices for the next 30 days are predicted, and the model can be improved by fine-tuning the architecture and hyperparameters.

## Kaggle Dataset Link:-https://www.kaggle.com/datasets/minatverma/nse-stocks-data/data

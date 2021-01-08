# LSTM forecast multi dimensional
## Stock prediction forecast with volume and news sentiment
We create an example script for forecasting of Time Series data. The example is based on following [LSTM article](https://stackabuse.com/solving-sequence-problems-with-lstm-in-keras/) in additional we add two dimensions Volume and News sentiment. News sentiment in the example data was tacken from our side [foreverycast.info](https://foreverycast.info/), where we aggregate daily news to a simple number in range between -1 (bad sentiment) to 1 (good sentiment). The idea is to create better forecasting data based not only on one dimension.

The multi dimensional script can create a forecast for set time period. The prediction for t+1 values is always based on the previous values. To create a prediction for values more than t+1, the previous will be used. In the prediction for t+2, first the prediction for t+1 will be done. As it is hard to predict news sentiment the value for t+1 is set to 0, same for volume dimension.

### Run the script
```shell
python3 lstm_many_to_one.py
```
Before the script starts, three variables can be set:
- File name, the name of the file with the data for learning.
- Backdays, a variable of int type. The variable determines how many "days" are relevant for the one future predicted value. (Default: 20)
- forecast_period, a variable of int type, The variable set the number of predicted "days". (Default: 10)

## To-Do
- Add type check for input data
- Add requirements.txt
- Add test.py

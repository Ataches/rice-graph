from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models import GraphData
import pandas as pd

from pmdarima import auto_arima
from math import isnan
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse

def predict_yield(df_saldana_harvest, lower_row_quantity):
    df_saldana_harvest = df_saldana_harvest[['Harvest_Date', 'Yield']]
    df_saldana_harvest.sort_values(by='Harvest_Date', inplace=True)
    df_saldana_harvest.set_index("Harvest_Date", inplace=True)
    print(auto_arima(df_saldana_harvest['Yield'], seasonal=True).summary())

    train_size = int(len(df_saldana_harvest) * 0.85)
    test_size = int(len(df_saldana_harvest)) - train_size

    train = df_saldana_harvest[:train_size]
    test = df_saldana_harvest[train_size:]


    arima_model = SARIMAX(train['Yield'])
    arima_result = arima_model.fit()
    forecast_yield = arima_result.forecast(steps=test_size, exog=test['Yield'])

    forecast_yield = pd.DataFrame(forecast_yield)
    forecast_yield.reset_index(drop=True, inplace=True)
    forecast_yield.index = test.index
    forecast_yield['Actual'] = df_saldana_harvest[train_size:]
    forecast_yield.rename(columns={'predicted_mean': 'Prediccion'}, inplace=True)

    forecast_yield['Actual'].plot(legend=True, color='blue')
    forecast_yield['Prediccion'].plot(legend=True, color='red')
    plt.show()
    error = rmse(forecast_yield['Prediccion'], forecast_yield['Actual'])
    print(error)

    print(arima_result.summary())

    predicted_line_set: dict = {}

    for year_to_predict in range(1, 4):
        pred_dynamic = arima_result.get_forecast(steps=lower_row_quantity * year_to_predict)
        forecast_3 = pred_dynamic.predicted_mean.to_numpy()[-1]
        predicted_line_set[str(2013 + year_to_predict)] = round(forecast_3, 2)

    return predicted_line_set


class GraphProcessing:

    def __init__(self):
        # Dataset load
        self.df_productivity = pd.read_csv('../dataset/1. DATA_Delerce2016_Saldana.csv', sep=',').dropna()
        self.df_historic_rain = pd.read_csv('../dataset/21135020_raw_prec.csv', sep=',').dropna()
        self.df_historic_rhum = pd.read_csv('../dataset/21135020_raw_rhum.csv', sep=',').dropna()
        self.df_historic_tmax = pd.read_csv('../dataset/21135020_raw_tmax.csv', sep=',').dropna()
        self.df_historic_tmin = pd.read_csv('../dataset/21135020_raw_tmin.csv', sep=',').dropna()
        self.df_historic_sbright = pd.read_csv('../dataset/21135020_raw_sbright.csv', sep=',').dropna()

        self.df_data_preparation()

    def df_data_preparation(self):
        self.df_productivity = self.df_productivity[['Municipality', 'Variety', 'Sowing_Date', 'Harvest_Date', 'Yield']]

        self.df_productivity['Harvest_Date'] = pd.to_datetime(self.df_productivity['Harvest_Date'])
        self.df_productivity['Sowing_Date'] = pd.to_datetime(self.df_productivity['Sowing_Date'])
        self.df_productivity = self.df_productivity[~self.df_productivity['Harvest_Date'].dt.date.duplicated()]

        self.df_productivity_saldana = self.df_productivity[(self.df_productivity['Municipality'] == 'SALDANA')]

        self.df_historic_rain['Date'] = pd.to_datetime(self.df_historic_rain['Date'], format="%Y%m%d")
        self.df_historic_rhum['Date'] = pd.to_datetime(self.df_historic_rhum['Date'], format="%Y%m%d")
        self.df_historic_tmax['Date'] = pd.to_datetime(self.df_historic_tmax['Date'], format="%Y%m%d")
        self.df_historic_tmin['Date'] = pd.to_datetime(self.df_historic_tmin['Date'], format="%Y%m%d")
        self.df_historic_sbright['Date'] = pd.to_datetime(self.df_historic_sbright['Date'], format="%Y%m%d")

        self.df_rain_by_variety = pd.DataFrame(columns=['Date', 'Value'])
        self.df_rhum_by_variety = pd.DataFrame(columns=['Date', 'Value'])
        self.df_tmax_by_variety = pd.DataFrame(columns=['Date', 'Value'])
        self.df_tmin_by_variety = pd.DataFrame(columns=['Date', 'Value'])
        self.df_sbright_by_variety = pd.DataFrame(columns=['Date', 'Value'])

    def df_by_bimester(self, year):
        better_yield: dict = {}

        better_yield["primer"] = round(self.df_by_variety[
            (self.df_by_variety['Harvest_Date'].dt.month < 4) & (self.df_by_variety['Harvest_Date'].dt.year == year)]['Yield'].mean(), 2)
        better_yield["segundo"] = round(self.df_by_variety[
            (self.df_by_variety['Harvest_Date'].dt.month > 3) & (self.df_by_variety['Harvest_Date'].dt.month < 7) & (
                    self.df_by_variety['Harvest_Date'].dt.year == year)]['Yield'].mean(), 2)
        better_yield["tercer"] = round(self.df_by_variety[
            (self.df_by_variety['Harvest_Date'].dt.month > 6) & (self.df_by_variety['Harvest_Date'].dt.month < 10) & (
                    self.df_by_variety['Harvest_Date'].dt.year == year)]['Yield'].mean(), 2)
        better_yield["cuarto"] = round(self.df_by_variety[
            (self.df_by_variety['Harvest_Date'].dt.month > 9) & (self.df_by_variety['Harvest_Date'].dt.year == year)]['Yield'].mean(), 2)
        return max({key: value for key, value in better_yield.items() if isnan(better_yield.get(key)) == False}.items(), key=lambda k: k[1])

    def graph_data_by_variety(self, msg):
        self.df_by_variety = self.df_productivity_saldana[
            self.df_productivity_saldana['Variety'] == msg['variety']]

        for index, row in self.df_by_variety.iterrows():
            self.df_rain_by_variety = pd.concat([(self.df_historic_rain[
                self.df_historic_rain['Date'].isin(pd.date_range(row['Sowing_Date'], row['Harvest_Date']))]),
                                                 self.df_rain_by_variety], ignore_index=True)
            self.df_rhum_by_variety = pd.concat([(self.df_historic_rhum[
                self.df_historic_rhum['Date'].isin(pd.date_range(row['Sowing_Date'], row['Harvest_Date']))]),
                                                 self.df_rhum_by_variety], ignore_index=True)
            self.df_tmax_by_variety = pd.concat([(self.df_historic_tmax[
                self.df_historic_tmax['Date'].isin(pd.date_range(row['Sowing_Date'], row['Harvest_Date']))]),
                                                 self.df_tmax_by_variety], ignore_index=True)
            self.df_tmin_by_variety = pd.concat([(self.df_historic_tmin[
                self.df_historic_tmin['Date'].isin(pd.date_range(row['Sowing_Date'], row['Harvest_Date']))]),
                                                 self.df_tmin_by_variety], ignore_index=True)
            self.df_sbright_by_variety = pd.concat([(self.df_historic_sbright[
                self.df_historic_sbright['Date'].isin(pd.date_range(row['Sowing_Date'], row['Harvest_Date']))]),
                                                    self.df_sbright_by_variety], ignore_index=True)

        years: list = self.df_by_variety['Harvest_Date'].dt.year.sort_values().unique()[-3:].tolist()
        rows_by_year: dict = {}
        graph_data: dict = {}
        yield_data: dict = {}
        lower_row_quantity = 10_000
        for i in years:
            rows_by_year[i] = self.df_by_variety[(self.df_by_variety['Harvest_Date'].dt.year == i)]
            row_quantity = (len(rows_by_year[i]))
            if 11 < row_quantity < lower_row_quantity:
                lower_row_quantity = row_quantity
            elif row_quantity <= 10:
                rows_by_year.pop(i)

        line_set: dict = {}
        df_saldana_harvest = pd.DataFrame()

        for i in rows_by_year.keys():
            line_set[str(i)] = round(
                self.df_by_variety[(self.df_by_variety['Harvest_Date'].dt.year == i)][:lower_row_quantity]['Yield'].mean(), 2)
            df_saldana_harvest = pd.concat([df_saldana_harvest,
                                            self.df_by_variety[self.df_by_variety['Harvest_Date'].dt.year == i]
                                            [:lower_row_quantity]])
            yield_data[i] = self.df_by_bimester(i)

        graph_data["lineSet"] = line_set
        graph_data["min_temp"] = round(self.df_tmin_by_variety['Value'].mean(), 2)
        graph_data["max_temp"] = round(self.df_tmax_by_variety['Value'].mean(), 2)
        graph_data["rhum"] = round(self.df_rhum_by_variety['Value'].mean(), 2)
        graph_data["sbright"] = round(self.df_sbright_by_variety['Value'].mean(), 2)
        graph_data["prec"] = round(self.df_rain_by_variety['Value'].mean(), 2)
        graph_data["prod"] = round(self.df_by_variety['Yield'].mean(), 2)
        graph_data["prec"] = round(self.df_rain_by_variety['Value'].mean(), 2)
        if len(line_set) > 1:
            graph_data["predictedLineSet"] = predict_yield(df_saldana_harvest, lower_row_quantity)
            graph_data["yield_data"] = max(yield_data.values(), key=lambda k: k[1])
        else:
            graph_data["predictedLineSet"] = {"": 0}
            graph_data["yield_data"] = self.df_by_bimester(years[len(years)-1])

        return GraphData(graph_data)

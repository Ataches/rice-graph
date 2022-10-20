from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models import GraphData
import pandas as pd

from pmdarima import auto_arima
from math import isnan
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.seasonal import seasonal_decompose


def predict_yield(df_saldana_harvest, lower_row_quantity):
    df_saldana_harvest = df_saldana_harvest[['Harvest_Date', 'Yield']]
    df_saldana_harvest.sort_values(by='Harvest_Date', inplace=True)
    df_saldana_harvest.set_index("Harvest_Date", inplace=True)

    df_saldana_harvest = df_saldana_harvest.resample('5D').mean()
    df_saldana_harvest = df_saldana_harvest.interpolate()

    df_saldana_harvest.asfreq('5D')

    decompose = seasonal_decompose(df_saldana_harvest)
    decompose.plot()
    plt.show()
    decompose.seasonal.plot(figsize=(16, 8))
    plt.show()

    train_size = int(len(df_saldana_harvest) * 0.85)
    test_size = int(len(df_saldana_harvest)) - train_size

    test = df_saldana_harvest[train_size:]

    # print(auto_arima(df_saldana_harvest, m=12, trace=True).summary())  # Used to get sarimax model

    arima_model = SARIMAX(df_saldana_harvest, order=(1, 0, 0))
    arima_result = arima_model.fit()
    prediction = arima_result.predict(train_size, (train_size+test_size-1)).rename('Prediction')
    test.plot(legend=True, figsize=(16, 8))
    prediction.plot(legend=True)
    plt.show()

    print("Raíz cuadrada media: ", rmse(test['Yield'], prediction))

    arima_result.plot_diagnostics(figsize=(16, 8))
    plt.show()

    steps_forecast = int(365/5)

    fig, ax = plt.subplots(figsize=(15, 5))

    arima_forecast = arima_result.get_forecast(steps=(steps_forecast*3)).summary_frame()
    df_saldana_harvest.plot(ax=ax)
    arima_forecast['mean'].plot(ax=ax, style='k--')
    ax.fill_between(arima_forecast.index, arima_forecast['mean_ci_lower'], arima_forecast['mean_ci_upper'], color='k', alpha=0.1);
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Producción')
    plt.legend()
    plt.show()

    predicted_line_set: dict = {}

    for year_to_predict in range(1, 4):
        forecast_mean = arima_forecast[:steps_forecast*year_to_predict].mean()['mean']
        predicted_line_set[str(2013 + year_to_predict)] = round(forecast_mean, 2)

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
                                           (self.df_by_variety['Harvest_Date'].dt.month < 4) & (
                                                       self.df_by_variety['Harvest_Date'].dt.year == year)][
                                           'Yield'].mean(), 2)
        better_yield["segundo"] = round(self.df_by_variety[
                                            (self.df_by_variety['Harvest_Date'].dt.month > 3) & (
                                                        self.df_by_variety['Harvest_Date'].dt.month < 7) & (
                                                    self.df_by_variety['Harvest_Date'].dt.year == year)][
                                            'Yield'].mean(), 2)
        better_yield["tercer"] = round(self.df_by_variety[
                                           (self.df_by_variety['Harvest_Date'].dt.month > 6) & (
                                                       self.df_by_variety['Harvest_Date'].dt.month < 10) & (
                                                   self.df_by_variety['Harvest_Date'].dt.year == year)]['Yield'].mean(),
                                       2)
        better_yield["cuarto"] = round(self.df_by_variety[
                                           (self.df_by_variety['Harvest_Date'].dt.month > 9) & (
                                                       self.df_by_variety['Harvest_Date'].dt.year == year)][
                                           'Yield'].mean(), 2)
        return max({key: value for key, value in better_yield.items() if isnan(better_yield.get(key)) == False}.items(),
                   key=lambda k: k[1])

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
                self.df_by_variety[(self.df_by_variety['Harvest_Date'].dt.year == i)][:lower_row_quantity][
                    'Yield'].mean(), 2)
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
            graph_data["yield_data"] = self.df_by_bimester(years[len(years) - 1])

        return GraphData(graph_data)

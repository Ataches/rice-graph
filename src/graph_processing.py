from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models import GraphData
import pandas as pd

from pmdarima import auto_arima


def predict_yield(df_saldana_harvest, line_set, lower_row_quantity):
    df_saldana_harvest = df_saldana_harvest[['Harvest_Date', 'Yield']]
    df_saldana_harvest.sort_values(by='Harvest_Date', inplace=True)
    df_saldana_harvest.set_index("Harvest_Date", inplace=True)
    print(auto_arima(df_saldana_harvest['Yield'], seasonal=True).summary())

    test_rows = int(len(df_saldana_harvest) * .30)
    train_data = df_saldana_harvest[:len(df_saldana_harvest) - test_rows]

    arima_model = SARIMAX(train_data['Yield'])
    arima_result = arima_model.fit()
    print(arima_result.summary())

    for year_to_predict in range(1, 5):
        pred_dynamic = arima_result.get_forecast(steps=lower_row_quantity * year_to_predict)
        forecast_3 = pred_dynamic.predicted_mean.to_numpy()[-1]
        line_set[str(2013 + year_to_predict)] = round(forecast_3, 2)

    return line_set


class GraphProcessing:

    def __init__(self):
        # Dataset load
        self.df_productivity = pd.read_csv('../dataset/1. DATA_Delerce2016_Saldana.csv', sep=',').dropna()
        self.df_historic_rain = pd.read_csv('../dataset/21135020_raw_prec.csv', sep=',').dropna()
        self.df_historic_rhum = pd.read_csv('../dataset/21135020_raw_rhum.csv', sep=',').dropna()
        self.df_historic_tmax = pd.read_csv('../dataset/21135020_raw_tmax.csv', sep=',').dropna()
        self.df_historic_tmin = pd.read_csv('../dataset/21135020_raw_tmin.csv', sep=',').dropna()
        self.df_historic_sbright = pd.read_csv('../dataset/21135020_raw_sbright.csv', sep=',').dropna()

        self.df_productivity = self.df_productivity[['Municipality', 'Variety', 'Sowing_Date', 'Harvest_Date', 'Yield']]

        self.df_productivity['Harvest_Date'] = pd.to_datetime(self.df_productivity['Harvest_Date'])
        self.df_productivity['Sowing_Date'] = pd.to_datetime(self.df_productivity['Sowing_Date'])
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

    def graph_data_by_variety(self, msg):
        df_by_variety = self.df_productivity_saldana[
            self.df_productivity_saldana['Variety'] == msg['variety']]

        for index, row in df_by_variety.iterrows():
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
        df_by_variety['Sowing_Date'].dt.year.sort_values().unique()[3:].tolist()
        years: list = df_by_variety['Sowing_Date'].dt.year.sort_values().unique()[3:].tolist()
        rows_by_year: dict = {}
        graph_data: dict = {}
        lower_row_quantity = 10_000
        for i in years:
            rows_by_year[i] = df_by_variety[
                (df_by_variety['Harvest_Date'].dt.year == i)]
            row_quantity = (len(rows_by_year[i]))
            if 11 < row_quantity < lower_row_quantity:
                lower_row_quantity = row_quantity
            elif row_quantity <= 10:
                rows_by_year.pop(i)

        line_set: dict = {}
        df_saldana_harvest = pd.DataFrame()

        for i in rows_by_year.keys():
            line_set[str(i)] = round(
                df_by_variety[(df_by_variety['Harvest_Date'].dt.year == i)][:lower_row_quantity]['Yield'].mean(), 2)
            df_saldana_harvest = pd.concat([df_saldana_harvest,
                                            df_by_variety[df_by_variety['Harvest_Date'].dt.year == i][
                                            :lower_row_quantity]])

        graph_data["lineSet"] = predict_yield(df_saldana_harvest, line_set, lower_row_quantity)
        graph_data["min_temp"] = round(self.df_tmin_by_variety['Value'].mean(), 2)
        graph_data["max_temp"] = round(self.df_tmax_by_variety['Value'].mean(), 2)
        graph_data["rhum"] = round(self.df_rhum_by_variety['Value'].mean(), 2)
        graph_data["sbright"] = round(self.df_sbright_by_variety['Value'].mean(), 2)
        graph_data["prec"] = round(self.df_rain_by_variety['Value'].mean(), 2)
        graph_data["prod"] = round(df_by_variety['Yield'].mean(), 2)

        return GraphData(graph_data)

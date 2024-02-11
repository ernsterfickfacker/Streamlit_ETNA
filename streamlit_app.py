import pandas as pd
import streamlit as st
from etna.datasets import generate_periodic_df
from etna.datasets import TSDataset
from etna.models import CatBoostPerSegmentModel
from etna.transforms import LagTransform
from etna.analysis import plot_forecast
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.analysis import plot_backtest
from etna.pipeline import Pipeline
max_horizon = 14 #2 weeks
st.title("Abrosimova Maria. Test Task")
st.markdown("""
<style>
.small-font {
    font-size:14px !important;
}
</style>
""", unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)


#table with task and graphics
col1, col2 = st.columns(2)
with col1:
   st.subheader("Task")
   st.markdown('<p class="small-font"> 1. Get familiar with ETNA time-series library and concept of time-series back-testing. </p>', unsafe_allow_html=True)
   st.markdown('<p class="small-font"> 2. Use CatBoostPerSegment and pipeline to build and validate your model. You basically need to use Get Started of the library. </p>',unsafe_allow_html=True)
   st.markdown('<p class="small-font"> 3. Build very simple Streamlit app where user can train and validate model. Use transforms of your choice. </p>',unsafe_allow_html=True)
   st.markdown('<p class="small-font"> 4. Visualize the results of model backtest and forecasts in the app.</p>',unsafe_allow_html=True)
   st.markdown('<p class="small-font"> 5. Visualize the results of model backtest and forecasts in the app.</p>',unsafe_allow_html=True)
with col2:
   st.subheader("Time Series examples")
   st.image("https://ernestoramirez.com/post/2016-02-23-visualizing-time-series-data_files/figure-html/minute_path_facet-1.png")
   st.image("https://ernestoramirez.com/post/2016-02-23-visualizing-time-series-data_files/figure-html/alldays_lineplot-1.png")

#radio group
dataset = st.radio("Choose DataSet", ('example_dataset', 'generated_periodic_dataset'))
if dataset == 'example_dataset':
    url = 'https://raw.githubusercontent.com/tinkoff-ai/etna/master/examples/data/example_dataset.csv'
    df_buf = pd.read_csv(url)
    #df = pd.read_csv(url, index_col=0, parse_dates=[0])
    max_horizon = 28
else:
        max_horizon = 14
        df_buf = generate_periodic_df(
        periods=100,
        start_time="2020-01-01",
        n_segments=4,
        period=7,
        sigma=3
    )

# Create a TSDataset
df = TSDataset.to_dataset(df=df_buf)
ts = TSDataset(df, freq="D")

#Visualization of time series
button1 = st.columns(5)[2].button("Show data and time-series 📈 " )
if button1:
    # visualize time-series
    st.markdown('<p class="small-font"> DataFrame </p>',unsafe_allow_html=True)
    st.dataframe(df)
    st.markdown('<p class="small-font"> Time-series </p>',unsafe_allow_html=True)
    st.pyplot(ts.plot(), clear_figure=None)
    print(ts.info())
    st.markdown('<p class="small-font"> The basic information about the dataset </p>', unsafe_allow_html=True)
    st.dataframe(ts.describe())

# Choose a horizon, the number of days we forecast for
horizon  = st.number_input('Choose a horizon', 1,max_horizon, value = 14)
#Splitting the data
train_ts, test_ts = ts.train_test_split(test_size=horizon)

# Prepare transforms
transforms = [
    LagTransform(in_column="target", lags=[horizon, horizon+1, horizon+2])
]
button2 = st.columns(5)[2].button("Build & validate model 📉" )
if button2:
    ts.fit_transform(transforms=transforms)
    future = ts.make_future(horizon, transforms=transforms)
    # Prepare model
    model = CatBoostPerSegmentModel()
    model.fit(ts=ts)
    CatBoostPerSegmentModel(iterations = None, depth = None, learning_rate = None,
    logging_level = 'Silent', l2_leaf_reg = None, thread_count = None, )
    forecast = model.forecast(future)
    forecast.inverse_transform(transforms)
    #pd.options.display.float_format = '{:,.2f}'.format
    #st.dataframe(forecast[:, :, "target"])

    # Create and fit the pipeline
    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
    pipeline.fit(train_ts)
    # Make a forecast
    forecast_ts = pipeline.forecast()
    #forecast interval
    fig2 = plot_forecast(forecast_ts=forecast_ts, test_ts=test_ts, train_ts=train_ts , n_train_samples=50, prediction_intervals=True)# train_ts=train_ts,
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown('<p class="small-font"> Forecast table </p>', unsafe_allow_html=True)
    st.dataframe(forecast_ts[:, :, "target"])
    st.markdown('<p class="small-font"> Forecast visualization </p>', unsafe_allow_html=True)
    st.pyplot(fig=fig2, clear_figure=None)#, use_container_width=True

    st.markdown( '<p class="small-font"> When constructing a forecast using Models and further evaluating the prediction metrics, we measure the quality at one time interval, designated as test. </p>', unsafe_allow_html=True)
    st.markdown( '<p class="small-font"> ▶ selects a period of time in the past </p>',unsafe_allow_html=True)
    st.markdown('<p class="small-font"> ▶ builds a model using the selected interval as a training sample </p>',unsafe_allow_html=True)
    st.markdown('<p class="small-font"> ▶ predicts the value on the test interval and calculates metrics </p>',unsafe_allow_html=True)

    #Backtest, metrics
    metrics_df, forecast_df, fold_info_df = pipeline.backtest(ts=ts, metrics=[MAE(), MSE(), SMAPE()],aggregate_metrics=True)
    st.markdown('<p class="small-font"> Metrics </p>', unsafe_allow_html=True)
    st.dataframe(metrics_df.head())
    st.markdown('<p class="small-font"> Backtest visualisation </p>',unsafe_allow_html=True)
    fig3 = plot_backtest(forecast_df, ts, history_len=50)
    st.pyplot(fig=fig3, clear_figure=None)

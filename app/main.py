import streamlit as st
import pickle
import os

# set title
st.title('Overall Player Rating Predictor')
st.subheader("Accuracy metrics", divider="rainbow")

# show accuracy metrics
col1, col2, col3, col4 = st.columns(spec=[.24, .24, .20, .30])
col1.metric("Mean absolute error", "1.20")
col2.metric("Mean squared error", "2.98")
col3.metric("R-squared score", "0.94")
col4.metric("Mean absolute percentage error", "0.019")
st.subheader("", divider="rainbow")

# load model
model = pickle.load(
    open('/mount/src/01_sportsprediction/app/model_xgb.pkl', 'rb'))


def predict(
        potential, wage_eur, passing,
        dribbling, attacking_short_passing,
        movement_reactions, power_shot_power,
        mentality_vision, mentality_composure):
    '''
    Predicts overall player rating based on user inputs

    args:
    potential (float): potential rating
    wage_eur (float): wage in EUR
    passing (float): passing rating
    dribbling (float): dribbling rating
    attacking_short_passing (float): attacking short passing rating
    movement_reactions (float): movement reactions rating
    power_shot_power (float): power shot power rating
    mentality_vision (float): mentality vision rating
    mentality_composure (float): mentality composure rating

    returns:
    overall rating (float): predicted overall rating
    '''
    prediction = model.predict([[
        potential, wage_eur,  passing,
        dribbling, attacking_short_passing,
        movement_reactions, power_shot_power,
        mentality_vision, mentality_composure
    ]])
    st.success("Overall rating is {:.2f}".format(prediction[0],))


# declare user inputs
potential = st.number_input(
    "Potential rating",
    key="potential",
    max_value=100.0,
    min_value=0.0
)

wage_eur = st.number_input(
    "Wage (in EUR)",
    key="wage_eur",
    min_value=0.0

)

passing = st.number_input(
    "Passing rating",
    key="passing",
    max_value=100.0,
    min_value=0.0

)

dribbling = st.number_input(
    "Dribbling rating",
    key="dribbling",
    max_value=100.0,
    min_value=0.0

)

attacking_short_passing = st.number_input(
    "Attacking short passing rating",
    key="attacking_short_passing",
    max_value=100.0,
    min_value=0.0

)

movement_reactions = st.number_input(
    "Movement reactions rating",
    key="movement_reactions",
    max_value=100.0,
    min_value=0.0

)

power_shot_power = st.number_input(
    "Power shot power rating",
    key="power_shot_power",
    max_value=100.0,
    min_value=0.0

)

mentality_vision = st.number_input(
    "Mentality vision rating",
    key="mentality_vision",
    max_value=100.0,
    min_value=0.0

)

mentality_composure = st.number_input(
    "Mentality composure rating",
    key="mentality_composure",
    max_value=100.0,
    min_value=0.0

)


col1, col2 = st.columns(spec=[.9, .1])

with col1:
    if st.button("Predict player's rating"):
        predict(
            potential, wage_eur, passing,
            dribbling, attacking_short_passing,
            movement_reactions, power_shot_power,
            mentality_vision, mentality_composure
        )

with col2:
    if st.button("Clear"):
        st.session_state = {}

import streamlit as st
import pickle
import numpy as np

@st.cache(allow_output_mutation=True)
def get_models():
    model=pickle.load(open('LM.pkl','rb'))
    sc=pickle.load(open('scaler.pkl','rb'))
    return (sc,model)


def predict(sc,model,rm,tax,lstat,rad,age):
    if rm==0:
        return 0
    else:
        input_data = [rm,lstat,tax,rad,age]
        np_input_data = np.array(input_data).reshape(1,-1)
        scaled_data=sc.transform(np_input_data)
        result=model.predict(scaled_data)[0]
        return result


def alaki():
    print('hello world')

if __name__ =='__main__':
    rm = st.number_input('average number of rooms per dwelling:')
    lstat = st.number_input('% lower status of the population:')
    tax = st.number_input('full-value property-tax rate per $10,000:')
    rad = st.number_input('index of accessibility to radial highways:')
    age = st.number_input('proportion of owner-occupied units built prior to 1940')
    if st.button('Predict'):
        sc,model = get_models()
        result = predict(sc,model,rm,tax,lstat,rad,age)
        st.write(result)

    
    
    
    

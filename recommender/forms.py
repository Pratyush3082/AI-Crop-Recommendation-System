from django import forms

class PredictForm(forms.Form):
    N = forms.FloatField(min_value=0)
    P = forms.FloatField(min_value=0)
    K = forms.FloatField(min_value=0)
    temperature = forms.FloatField()
    humidity = forms.FloatField()
    ph = forms.FloatField()
    rainfall = forms.FloatField()
    season = forms.ChoiceField(choices=[('Kharif','Kharif'),('Rabi','Rabi'),('Zaid','Zaid')], required=False)
    location = forms.CharField(required=False)

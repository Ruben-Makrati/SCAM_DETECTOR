from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import ScamReport


class TextDetectionForm(forms.Form):
    """Form for text scam detection"""
    text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': 'Enter the text you want to check for scams...',
            'id': 'detection-text'
        }),
        label='Text to Analyze',
        max_length=1000,
        help_text='Enter the suspicious text, message, or offer you want to check.'
    )


class ScamReportForm(forms.ModelForm):
    """Form for reporting scams"""
    class Meta:
        model = ScamReport
        fields = ['report_type', 'description', 'contact_info']
        widgets = {
            'report_type': forms.Select(attrs={
                'class': 'form-control',
                'id': 'report-type'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe the scam in detail...',
                'id': 'report-description'
            }),
            'contact_info': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Phone number, email, or website (optional)',
                'id': 'report-contact'
            })
        }
        labels = {
            'report_type': 'Type of Scam',
            'description': 'Description',
            'contact_info': 'Contact Information (Optional)'
        }
        help_texts = {
            'description': 'Please provide as much detail as possible about the scam.',
            'contact_info': 'If you have contact information from the scammer, include it here.'
        }


class UserRegistrationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(max_length=254, required=True, widget=forms.EmailInput(attrs={'class': 'form-control'}))

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2') 
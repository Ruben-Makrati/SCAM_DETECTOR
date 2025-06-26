from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User


class DetectionHistory(models.Model):
    """Model to store detection history"""
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='detection_histories')
    text = models.TextField()
    is_scam = models.BooleanField()
    confidence_score = models.FloatField()
    detected_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    USER_FEEDBACK_CHOICES = [
        ('not_set', 'Not Set'),
        ('correct', 'Correct'),
        ('incorrect', 'Incorrect'),
    ]
    user_feedback = models.CharField(max_length=10, choices=USER_FEEDBACK_CHOICES, default='not_set')
    
    class Meta:
        verbose_name_plural = "Detection History"
        ordering = ['-detected_at']
    
    def __str__(self):
        return f"{'SCAM' if self.is_scam else 'LEGIT'} - {self.text[:50]}..."


class ScamReport(models.Model):
    """Model to store user-reported scams"""
    REPORT_TYPES = [
        ('email', 'Email Scam'),
        ('sms', 'SMS Scam'),
        ('phone', 'Phone Call Scam'),
        ('website', 'Website Scam'),
        ('social', 'Social Media Scam'),
        ('other', 'Other'),
    ]
    
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES)
    description = models.TextField()
    contact_info = models.CharField(max_length=255, blank=True)
    reported_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    is_verified = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-reported_at']
    
    def __str__(self):
        return f"{self.get_report_type_display()} - {self.description[:50]}..."


class ScamStatistics(models.Model):
    """Model to store scam detection statistics"""
    date = models.DateField(unique=True)
    total_detections = models.IntegerField(default=0)
    scam_detections = models.IntegerField(default=0)
    legitimate_detections = models.IntegerField(default=0)
    
    class Meta:
        verbose_name_plural = "Scam Statistics"
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.date} - {self.scam_detections}/{self.total_detections} scams"
    
    @property
    def scam_percentage(self):
        if self.total_detections == 0:
            return 0
        return (self.scam_detections / self.total_detections) * 100 
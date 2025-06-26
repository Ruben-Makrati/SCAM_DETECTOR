from django.contrib import admin
from .models import DetectionHistory, ScamReport, ScamStatistics


@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = ['text_preview', 'is_scam', 'confidence_score', 'detected_at', 'ip_address']
    list_filter = ['is_scam', 'detected_at']
    search_fields = ['text']
    readonly_fields = ['detected_at']
    ordering = ['-detected_at']
    
    def text_preview(self, obj):
        return obj.text[:100] + '...' if len(obj.text) > 100 else obj.text
    text_preview.short_description = 'Text'


@admin.register(ScamReport)
class ScamReportAdmin(admin.ModelAdmin):
    list_display = ['report_type', 'description_preview', 'contact_info', 'reported_at', 'is_verified']
    list_filter = ['report_type', 'reported_at', 'is_verified']
    search_fields = ['description', 'contact_info']
    readonly_fields = ['reported_at']
    ordering = ['-reported_at']
    
    def description_preview(self, obj):
        return obj.description[:100] + '...' if len(obj.description) > 100 else obj.description
    description_preview.short_description = 'Description'


@admin.register(ScamStatistics)
class ScamStatisticsAdmin(admin.ModelAdmin):
    list_display = ['date', 'total_detections', 'scam_detections', 'legitimate_detections', 'scam_percentage']
    list_filter = ['date']
    readonly_fields = ['scam_percentage']
    ordering = ['-date']
    
    def scam_percentage(self, obj):
        return f"{obj.scam_percentage:.1f}%"
    scam_percentage.short_description = 'Scam %' 
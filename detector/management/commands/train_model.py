from django.core.management.base import BaseCommand
from detector.ml_model import get_scam_detector


class Command(BaseCommand):
    help = 'Train the scam detection machine learning model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--csv-path',
            type=str,
            default='spam.csv',
            help='Path to the CSV file containing training data'
        )
        parser.add_argument(
            '--save-model',
            action='store_true',
            help='Save the trained model to disk'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting model training...')
        )
        
        try:
            # Get the scam detector instance
            detector = get_scam_detector()
            
            # Train the model
            accuracy = detector.train(csv_path=options['csv_path'])
            
            self.stdout.write(
                self.style.SUCCESS(f'Model training completed successfully!')
            )
            self.stdout.write(
                self.style.SUCCESS(f'Accuracy: {accuracy:.2%}')
            )
            
            # Save the model if requested
            if options['save_model']:
                detector.save_model()
                self.stdout.write(
                    self.style.SUCCESS('Model saved to disk successfully!')
                )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error training model: {str(e)}')
            ) 
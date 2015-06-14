from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from functions.gensim_json import MyCorpus
import time

class Command(BaseCommand):
    help = 'Initialize neccesary data'

    def handle(self, *args, **options):
        start_time = time.time()
        corpus = MyCorpus(settings.JSON_KEY)
        corpus.first_run()
        self.stdout.write("--- %s seconds ---" % (time.time() - start_time))
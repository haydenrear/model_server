import injector
from drools_py.classification_models.classify_spam import SpamClassifier
from drools_py.classification_models.torch_classification import StringClassifier
from python_di.env.init_env import import_load
from python_di.inject.injector_provider import InjectionContext
from injector import Binder

from metadata_extractor.delegating_metadata_extractor import MultiStringClassifier, DelegatingMetadataExtractor


class MetadataExtractorModules(injector.Module):
    def configure(self, binder: Binder) -> None:
        if InjectionContext.environment is None:
            InjectionContext.do_initialize()
        classifiers = InjectionContext.environment.get_property('classifiers')
        loaded_classifiers = []
        if isinstance(classifiers, list):
            for classifier in classifiers:
                try:
                    loaded_classifiers.append(import_load(classifier))
                except Exception as e:
                    print(f'Failed to load {classifiers}: {e}')
        else:
            print("No classifiers found.")

        binder.multibind(list[StringClassifier], loaded_classifiers)
        binder.bind(MultiStringClassifier, DelegatingMetadataExtractor)
        binder.bind(SpamClassifier, SpamClassifier)


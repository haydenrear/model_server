import os

import python_di.env.env_properties
import yaml


class ClassifierEnvironment(python_di.env.env_properties.YamlPropertiesFilesBasedEnvironment):

    def load_props_inner(self, join):
        super().load_props_inner(join)
        for file in os.listdir(join):
            if os.path.isfile(os.path.join(join, file)):
                with open(f"{os.path.join(join, file)}", "r") as props:
                    props = yaml.safe_load(props)
                    classifiers = []
                    if 'classifiers' in props.keys() and props['classifiers']:
                        for classifier in props['classifiers']:
                            classifiers.append(classifier)
                    self.set_property('classifiers', classifiers)

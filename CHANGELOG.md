# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.0] - 2021-05-27
### Changed
- Updated Metadata in setup.py
- Merged Implementations MrBuilder-PyTorch and MrBuilder-Keras into MrBuilder repository
- Keras Changelog
  
        ## [0.2.0] - 2021-01-29
        ### Changed
        - Move to functional layer registration to confirm with pytorch implementation
        
        
        ## [0.1.0] - 2020-02-04
        ### Added
        - Initial Version extracted from Core Implementation
        - Moved Test cases to implementation
        - Simplified library files
        - Extracted out implementation to separate files
- Pytorch Changelog
  
        ## [0.5.1] - 2021-05-03
        ### Changed
        - Move test utils to test utils folder
        
        ## [0.5] - 2021-04-20
        ### Added
        - Add ability to nest layers
        - Add forward dynamic params
        - Custom hardcoded templates
        ### Changed
        - Use links for input, add multi input ability
        - Updates for testing
        - Use smart naming convention for layers
        - Use links for output size
        - Use inline weight initiation
        - Add dynamic scanning of Pytorch Modules
        - Use MrBuilder global Tests
        ### Fixed
        - Hardcode Test VGG16 - 2021-04-21
        
        ## [0.0.5] - 2021-01-29
        ### Added
        - Initial Version

## [0.9.9] - 2021-04-21
### Added
- Add basic model args camel snake 
### Changed
- Add ability to use kwargs for model creation
### Fixed
- Model Kwargs vs param 


## [0.9.8] - 2021-03-29
### Changed
- Fixed Duplicate creation bug
- Moved Test Cases to Root Project
- Bug Fixes

## [0.9.5] - 2021-01-29
### Changed 
- Move internal code around
- Add Decorator for layer registration
- Add AliasedDict for that allows layers to have multiple names
- Add layer signature inspection
- Redo expression execution
- Fix variable find issues


## [0.9.0] - 2020-02-04
### Changed
- Moved Keras Implementation to separate Repo "mrbuilder_keras"
- Removed public implementation from Core Repo
- Moved Test Cases for implementation to Keras Repo
- Simplified naming for core methods
- Moved loader into core implementation
### Removed
- Singleton class / requirements

## [0.8.0] - 2019-08-05
### Added
- Added "If" Statements in builder
- Added boolean logic to expressions, and, or, >, <, ==, !=, added boolean conversion support  
- Updated Readme to reflect new Vgg16
### Removed
- Support for "--ignore" logic in layers

## [0.7.3] - 2019-07-23
### Changed
- Fixed expression recursion bug with name collisions
- Updated Library Vgg16 to use configurable params
- Updated Readme to reflect new Vgg16

## [0.7.1] - 2019-07-23
### Changed
- Fixed multi init bug

## [0.7.0] - 2019-07-22
### Added
- Support for nesting templates
- Support for expression to expression evaluation
- Model library tests
- Better expressions test coverage
- Add SqueezeNet to library, add tests for it
   
### Changed
- Separate out builder tests
- Extract out variable resolution
- Clean up model builder
- Change variable resolver naming convention
- Make expression parsing class based 

### Removed
- Support for inline variable resolve without expressions

## [0.6.0] - 2014-06-18
### Added
- Single field expression support + test cases
- Refactor builder internals
- Clean up internal naming conventions
- Clean up test data
- Vgg16 to model library

## [0.5.4] - 2019-06-14
### Changed
- Cleaned up .gitignore

## [0.5.3] - 2019-06-14
### Changed
- Fixed Circular Import
- Pushed to GitHub

## [0.5.2] - 2014-08-09
### Changed
- Documentation to reflect new built in layer xutils.

## [0.5.1] - 2019-01-26
### Added
- Readme documentation for standard models Vgg16

## [0.5.0] - 2019-01-26
### Added
- Initial Version
- Core Model Parser
- Core Library Functionality
- Simple test cases


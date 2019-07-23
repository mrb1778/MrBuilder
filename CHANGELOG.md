# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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
- Documentation to reflect new built in layer utils.

## [0.5.1] - 2019-01-26
### Added
- Readme documentation for standard models Vgg16

## [0.5.0] - 2019-01-26
### Added
- Initial Version
- Core Model Parser
- Core Library Functionality
- Simple test cases

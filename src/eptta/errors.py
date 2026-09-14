class EPTTAError(ValueError):
    exit_code = 2
    code = "INVALID_CONFIG"


class ContractError(EPTTAError):
    code = "BLOCKED_CONTRACT"


class PermissionDenied(EPTTAError):
    code = "PERMISSION_DENIED"


class MissingDependency(EPTTAError):
    code = "NOT_RUN_MISSING_DEPENDENCY"


class NotImplementedStage(EPTTAError):
    code = "NOT_IMPLEMENTED_STAGE"


class ResourceError(EPTTAError):
    exit_code = 3
    code = "MISSING_RESOURCE"


class DataError(EPTTAError):
    exit_code = 4
    code = "DATA_INCOMPLETE"

# This file contains tyope definitions

"""
    FieldType

Union of all concrete field types that share a common BroadcastStyle override:
`FTField`, `Field`, `VectorField`, and `ProjectedField`.

Used internally by the broadcasting hooks so that a single `BroadcastStyle`
registration covers all field types.
"""
const FieldType = Union{FTField,Field,VectorField,ProjectedField}
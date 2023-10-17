# Tech Debts

1. Centralized input system (can use `dataclass`):
   1. Single file for all inputs,
   2. Separate classes for models inherited from a structured enum,
   3. Separate class for the constants in the input file,
   4. Separate possible input configurations,
2. Take input for the driver code from an external json file instead of modifying the dictionary in the driver code each time.

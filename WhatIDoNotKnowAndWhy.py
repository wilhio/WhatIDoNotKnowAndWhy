import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sympy as sp
from typing import List, Dict, Tuple, Optional, Union, Any
from torchvision import datasets, transforms


# ======================================================
# 3-Valued Operators
# ======================================================
# --- Kleene's 3-valued logic (values: 0, 0.5, 1)
def kleene_not(x):
    return 1 - x

def kleene_and(x, y):
    return min(x, y)

def kleene_or(x, y):
    return max(x, y)

def kleene_imp(x, y):
    # Define implication as: if x <= y then 1, else y.
    return 1 if x <= y else y

def kleene_equiv(x, y):
    # Equivalence as the minimum of the two implications.
    return min(kleene_imp(x, y), kleene_imp(y, x))

# --- Lukasiewicz's 3-valued logic
def lukasiewicz_not(x):
    return 1 - x

def lukasiewicz_and(x, y):
    # T-norm: max(0, x+y-1)
    return max(0, x + y - 1)

def lukasiewicz_or(x, y):
    # S-norm: min(1, x+y)
    return min(1, x + y)

def lukasiewicz_imp(x, y):
    # Standard Lukasiewicz implication: min(1, 1 - x + y)
    return min(1, 1 - x + y)

def lukasiewicz_equiv(x, y):
    return min(lukasiewicz_imp(x, y), lukasiewicz_imp(y, x))


class MemoryPool:
    """
    A memory pool for storing and retrieving logical facts and inferences.
    """
    def __init__(self, capacity: int = 1000, pool_type: str = "neutral"):
        """
        Initialize a memory pool with a specified capacity.
        
        Args:
            capacity: Maximum number of items to store in the pool
            pool_type: Type of memory pool ("neutral", "a", or "b")
        """
        self.capacity = capacity
        self.memory = []
        self.access_counts = {}  # Track access frequency
        self.timestamps = {}     # Track recency
        self.current_time = 0
        self.pool_type = pool_type  # Identifies which logic system this pool belongs to
    
    def add(self, item: Any, metadata: Dict = None) -> bool:
        """
        Add an item to the memory pool.
        
        Args:
            item: The item to store
            metadata: Additional information about the item
            
        Returns:
            Boolean indicating success
        """
        if len(self.memory) >= self.capacity:
            self._evict()
            
        if item not in self.memory:
            self.memory.append(item)
            self.access_counts[item] = 0
            self.timestamps[item] = self.current_time
            self.current_time += 1
            return True
        return False
    
    def get(self, query: Any = None, filter_func: callable = None) -> List[Any]:
        """
        Retrieve items from memory based on a query or filter function.
        
        Args:
            query: Query to match against items
            filter_func: Function to filter items
            
        Returns:
            List of matching items
        """
        results = []
        
        if query is not None:
            for item in self.memory:
                if item == query or (hasattr(item, '__contains__') and query in item):
                    results.append(item)
                    self._update_access(item)
        
        elif filter_func is not None:
            for item in self.memory:
                if filter_func(item):
                    results.append(item)
                    self._update_access(item)
        
        else:
            # Return all items if no query or filter provided
            results = self.memory.copy()
            for item in results:
                self._update_access(item)
                
        return results
    
    def _update_access(self, item: Any) -> None:
        """
        Update access statistics for an item.
        
        Args:
            item: The item being accessed
        """
        if item in self.access_counts:
            self.access_counts[item] += 1
            self.timestamps[item] = self.current_time
            self.current_time += 1
    
    def _evict(self) -> None:
        """
        Evict an item from memory based on access frequency and recency.
        """
        if not self.memory:
            return
            
        # Compute a score based on frequency and recency
        scores = {}
        for item in self.memory:
            frequency_score = self.access_counts[item]
            recency_score = self.timestamps[item]
            scores[item] = 0.7 * frequency_score + 0.3 * recency_score
            
        # Find the item with the lowest score
        item_to_evict = min(scores, key=scores.get)
        
        # Remove the item
        self.memory.remove(item_to_evict)
        del self.access_counts[item_to_evict]
        del self.timestamps[item_to_evict]
    
    def clear(self) -> None:
        """
        Clear all items from memory.
        """
        self.memory = []
        self.access_counts = {}
        self.timestamps = {}
        self.current_time = 0
    
    def __len__(self) -> int:
        """
        Get the current number of items in memory.
        
        Returns:
            Integer count of items
        """
        return len(self.memory)


class LogicBehavior:
    """
    Defines a specific logic behavior pattern that can be applied to a logic system.
    """
    def __init__(self, name: str, description: str, logic_type: str = "kleene"):
        """
        Initialize a logic behavior.
        
        Args:
            name: Name of the behavior
            description: Description of how this behavior works
            logic_type: Type of logic to use ("kleene" or "lukasiewicz")
        """
        self.name = name
        self.description = description
        self.rules = []
        self.inference_strategy = "forward"  # Can be "forward", "backward", or "bidirectional"
        self.conflict_resolution = "priority"  # How to resolve conflicting rules
        self.logic_type = logic_type
        
    def add_rule(self, condition: str, consequence: str, priority: int = 0) -> None:
        """
        Add a rule to this behavior.
        
        Args:
            condition: Logical condition
            consequence: Result when condition is true
            priority: Rule priority (higher numbers have higher priority)
        """
        self.rules.append({
            "condition": condition,
            "consequence": consequence,
            "priority": priority
        })
        
    def set_inference_strategy(self, strategy: str) -> None:
        """
        Set the inference strategy for this behavior.
        
        Args:
            strategy: One of "forward", "backward", or "bidirectional"
        """
        if strategy in ["forward", "backward", "bidirectional"]:
            self.inference_strategy = strategy
            
    def set_conflict_resolution(self, method: str) -> None:
        """
        Set the conflict resolution method.
        
        Args:
            method: Method to use ("priority", "specificity", "recency")
        """
        if method in ["priority", "specificity", "recency"]:
            self.conflict_resolution = method
            
    def set_logic_type(self, logic_type: str) -> None:
        """
        Set the type of 3-valued logic to use.
        
        Args:
            logic_type: Type of logic ("kleene" or "lukasiewicz")
        """
        if logic_type in ["kleene", "lukasiewicz"]:
            self.logic_type = logic_type


class LogicSystemA(LogicBehavior):
    """
    Logic behavior type A: Deductive reasoning with strict rule application.
    """
    def __init__(self, logic_type: str = "kleene"):
        super().__init__(
            name="Deductive Logic",
            description="Applies rules strictly in a top-down manner, with emphasis on consistency and completeness.",
            logic_type=logic_type
        )
        self.set_inference_strategy("forward")
        self.set_conflict_resolution("priority")
        
        # Additional A-specific properties
        self.requires_consistency = True
        self.allows_uncertainty = True  # Now allows uncertainty with 3-valued logic
        self.truth_maintenance = "strict"
        
    def evaluate_expression(self, expression: str, variable_states: Dict[str, float]) -> float:
        """
        Evaluate a logical expression using 3-valued logic.
        
        Args:
            expression: Logical expression to evaluate
            variable_states: Current variable states with values in [0, 0.5, 1]
            
        Returns:
            Float result of the expression (0, 0.5, or 1)
        """
        # For simple expressions, implement 3-valued logic
        if "and" in expression:
            parts = expression.split("and")
            values = []
            for part in parts:
                part = part.strip()
                if part in variable_states:
                    values.append(variable_states[part])
            
            # Apply appropriate 3-valued AND operator
            if not values:
                return 0.0
            
            result = values[0]
            for val in values[1:]:
                if self.logic_type == "kleene":
                    result = kleene_and(result, val)
                else:  # lukasiewicz
                    result = lukasiewicz_and(result, val)
            return result
            
        elif "or" in expression:
            parts = expression.split("or")
            values = []
            for part in parts:
                part = part.strip()
                if part in variable_states:
                    values.append(variable_states[part])
            
            # Apply appropriate 3-valued OR operator
            if not values:
                return 0.0
                
            result = values[0]
            for val in values[1:]:
                if self.logic_type == "kleene":
                    result = kleene_or(result, val)
                else:  # lukasiewicz
                    result = lukasiewicz_or(result, val)
            return result
            
        elif "not" in expression:
            var = expression.replace("not", "").strip()
            if var in variable_states:
                if self.logic_type == "kleene":
                    return kleene_not(variable_states[var])
                else:  # lukasiewicz
                    return lukasiewicz_not(variable_states[var])
        
        # For a single variable
        if expression in variable_states:
            return variable_states[expression]
            
        return 0.0


class LogicSystemB(LogicBehavior):
    """
    Logic behavior type B: Inductive/probabilistic reasoning with fuzzy rule application.
    """
    def __init__(self, logic_type: str = "lukasiewicz"):
        super().__init__(
            name="Inductive Logic",
            description="Applies rules with degrees of certainty, allowing for partial matches and probabilistic inference.",
            logic_type=logic_type
        )
        self.set_inference_strategy("bidirectional")
        self.set_conflict_resolution("specificity")
        
        # Additional B-specific properties
        self.requires_consistency = False
        self.allows_uncertainty = True
        self.truth_maintenance = "fuzzy"
        self.certainty_threshold = 0.5  # Adjusted for 3-valued logic
        
    def evaluate_expression(self, expression: str, variable_states: Dict[str, float]) -> float:
        """
        Evaluate a logical expression using 3-valued logic with certainty factors.
        
        Args:
            expression: Logical expression to evaluate
            variable_states: Current variable states with values in [0, 1]
            
        Returns:
            Float representing certainty of the result (0.0 to 1.0)
        """
        # For simple expressions, implement 3-valued logic
        if "and" in expression:
            parts = expression.split("and")
            values = []
            for part in parts:
                part = part.strip()
                if part in variable_states:
                    values.append(variable_states[part])
            
            # Apply appropriate 3-valued AND operator
            if not values:
                return 0.0
                
            result = values[0]
            for val in values[1:]:
                if self.logic_type == "kleene":
                    result = kleene_and(result, val)
                else:  # lukasiewicz
                    result = lukasiewicz_and(result, val)
            return result
            
        elif "or" in expression:
            parts = expression.split("or")
            values = []
            for part in parts:
                part = part.strip()
                if part in variable_states:
                    values.append(variable_states[part])
            
            # Apply appropriate 3-valued OR operator
            if not values:
                return 0.0
                
            result = values[0]
            for val in values[1:]:
                if self.logic_type == "kleene":
                    result = kleene_or(result, val)
                else:  # lukasiewicz
                    result = lukasiewicz_or(result, val)
            return result
            
        elif "not" in expression:
            var = expression.replace("not", "").strip()
            if var in variable_states:
                if self.logic_type == "kleene":
                    return kleene_not(variable_states[var])
                else:  # lukasiewicz
                    return lukasiewicz_not(variable_states[var])
        
        # For a single variable
        if expression in variable_states:
            return variable_states[expression]
            
        return 0.0


class ArgumentationFramework:
    def __init__(self, arguments, attacks):
        """
        arguments: list of strings representing arguments
        attacks: list of tuples (attacker, target)
        """
        self.arguments = arguments
        self.attacks = attacks
        # Create a symbol for each argument.
        self.symbols = {arg: sp.symbols(arg) for arg in arguments}

    def attackers_of(self, arg):
        """Return the list of arguments that attack the given argument."""
        return [attacker for attacker, target in self.attacks if target == arg]

    def attackers_of_attack(self, arg):
        """
        For an argument 'arg' (which is an attacker of some other argument),
        return the list of arguments that attack 'arg'.
        """
        return [attacker for attacker, target in self.attacks if target == arg]

    def elementary_encoding(self, use_alternative=False):
        """
        Elementary encoding (ec0).

        Option 1 (default): For each argument a in A, encode as
            a -> (AND_{b attacks a} ¬b)
        Option 2 (alternative): Use the equivalent form
            AND_{(a,b) in R} (¬a OR ¬b)
        """
        if use_alternative:
            # Alternative encoding: ∧_{(a, b) in R} (¬a ∨ ¬b)
            formulas = [sp.Or(sp.Not(self.symbols[a]), sp.Not(self.symbols[b]))
                        for a, b in self.attacks]
            enc = sp.And(*formulas)
        else:
            # Standard elementary encoding: for each a, a -> (∧_{b attacks a} ¬b)
            formulas = []
            for a in self.arguments:
                attackers = self.attackers_of(a)
                # If no attacker, use True (empty And() returns True in sympy)
                if attackers:
                    not_attackers = [sp.Not(self.symbols[b]) for b in attackers]
                    body = sp.And(*not_attackers)
                else:
                    body = sp.true
                formulas.append(sp.Implies(self.symbols[a], body))
            enc = sp.And(*formulas)
        return sp.simplify_logic(enc, force=True)

    def normal_encoding(self):
        """
        Normal encoding (ec1):
            For each argument a in A, encode as
            a ↔ (∧_{b attacks a} ¬b)
        If a has no attackers, then the right-hand side is True, forcing a to be True.
        """
        formulas = []
        for a in self.arguments:
            attackers = self.attackers_of(a)
            if attackers:
                body = sp.And(*[sp.Not(self.symbols[b]) for b in attackers])
            else:
                body = sp.true
            formulas.append(sp.Equivalent(self.symbols[a], body))
        return sp.simplify_logic(sp.And(*formulas), force=True)

    def regular_encoding(self):
        """
        Regular encoding (ec2):
            For each argument a in A, encode as the conjunction of:
              1. a → (∧_{b attacks a} ¬b)
              2. a ↔ (∧_{b attacks a} (∨_{c attacks b} c))
        In the inner disjunction, if an attacker b has no attackers then by convention
        the empty disjunction is False.
        """
        formulas = []
        for a in self.arguments:
            attackers = self.attackers_of(a)
            # Part 1: a -> (∧_{b attacks a} ¬b)
            if attackers:
                part1_body = sp.And(*[sp.Not(self.symbols[b]) for b in attackers])
            else:
                part1_body = sp.true
            part1 = sp.Implies(self.symbols[a], part1_body)

            # Part 2: a ↔ (∧_{b attacks a} (∨_{c attacks b} c))
            # For each attacker b of a, get the disjunction of all attackers of b.
            disjuncts = []
            for b in attackers:
                attackers_b = self.attackers_of_attack(b)
                # By convention, an empty disjunction is False.
                if attackers_b:
                    disj = sp.Or(*[self.symbols[c] for c in attackers_b])
                else:
                    disj = sp.false
                disjuncts.append(disj)
            # If there are no attackers of a, then the conjunction is True.
            if disjuncts:
                part2_body = sp.And(*disjuncts)
            else:
                part2_body = sp.true
            part2 = sp.Equivalent(self.symbols[a], part2_body)

            formulas.append(sp.And(part1, part2))
        return sp.simplify_logic(sp.And(*formulas), force=True)


class LogicSystem:
    """
    A class for representing and manipulating logical systems with dual behavior modes.
    """
    def __init__(self, variables: List[str] = None, active_logic: str = "neutral"):
        """
        Initialize a logic system with variables and dual logic behaviors.
        
        Args:
            variables: List of variable names in the system
            active_logic: Which logic system to use by default ("neutral", "a", or "b")
        """
        self.variables = variables or []
        self.variable_states = {var: 0.0 for var in self.variables}  # Initialize with 0 (false)
        self.truth_table = {}
        
        # Create separate memory pools for each logic system
        self.memory_pool_neutral = MemoryPool(capacity=1000, pool_type="neutral")
        self.memory_pool_a = MemoryPool(capacity=500, pool_type="a")
        self.memory_pool_b = MemoryPool(capacity=500, pool_type="b")
        
        # Initialize the two logic behaviors with 3-valued logic
        self.logic_a = LogicSystemA(logic_type="kleene")
        self.logic_b = LogicSystemB(logic_type="lukasiewicz")
        
        # Set the active logic system
        self.active_logic = active_logic
        self.inference_history = []
        
        # Initialize argumentation framework
        self.af = None
        
    def add_variable(self, variable: str) -> None:
        """
        Add a new variable to the logic system.
        
        Args:
            variable: Name of the variable to add
        """
        if variable not in self.variables:
            self.variables.append(variable)
            self.variable_states[variable] = 0.0  # Initialize with 0 (false)
            
            # Add to all memory pools
            self.memory_pool_neutral.add(f"Variable added: {variable}")
            self.memory_pool_a.add(f"Variable added: {variable}")
            self.memory_pool_b.add(f"Variable added: {variable}")
            
    def remove_variable(self, variable: str) -> None:
        """
        Remove a variable from the logic system.
        
        Args:
            variable: Name of the variable to remove
        """
        if variable in self.variables:
            self.variables.remove(variable)
            del self.variable_states[variable]
            
            # Remove rules that reference this variable from both logic systems
            self.logic_a.rules = [rule for rule in self.logic_a.rules 
                               if variable not in rule.get('condition', '') 
                               and variable not in rule.get('consequence', '')]
            
            self.logic_b.rules = [rule for rule in self.logic_b.rules 
                               if variable not in rule.get('condition', '') 
                               and variable not in rule.get('consequence', '')]
            
            # Add to all memory pools
            self.memory_pool_neutral.add(f"Variable removed: {variable}")
            self.memory_pool_a.add(f"Variable removed: {variable}")
            self.memory_pool_b.add(f"Variable removed: {variable}")
    
    def add_rule_to_a(self, condition: str, consequence: str, priority: int = 0) -> None:
        """
        Add a logical rule to system A.
        
        Args:
            condition: Logical condition (e.g., "A and B")
            consequence: Result when condition is true (e.g., "C")
            priority: Rule priority
        """
        self.logic_a.add_rule(condition, consequence, priority)
        self.memory_pool_a.add(f"Rule added to A: IF {condition} THEN {consequence}", 
                            {"type": "rule", "logic": "a", "condition": condition, "consequence": consequence})
    
    def add_rule_to_b(self, condition: str, consequence: str, priority: int = 0) -> None:
        """
        Add a logical rule to system B.
        
        Args:
            condition: Logical condition (e.g., "A and B")
            consequence: Result when condition is true (e.g., "C")
            priority: Rule priority
        """
        self.logic_b.add_rule(condition, consequence, priority)
        self.memory_pool_b.add(f"Rule added to B: IF {condition} THEN {consequence}", 
                            {"type": "rule", "logic": "b", "condition": condition, "consequence": consequence})
    
    def set_variable(self, variable: str, value: float) -> None:
        """
        Set the truth value of a variable.
        
        Args:
            variable: Name of the variable
            value: Truth value (0.0, 0.5, or 1.0 for 3-valued logic)
        """
        if variable in self.variable_states:
            old_value = self.variable_states[variable]
            # Ensure value is in [0, 0.5, 1] for 3-valued logic
            if value < 0.25:
                value = 0.0
            elif value < 0.75:
                value = 0.5
            else:
                value = 1.0
                
            self.variable_states[variable] = value
            
            # Record in appropriate memory pools
            if self.active_logic == "a":
                self.memory_pool_a.add(f"Variable {variable} changed from {old_value} to {value}",
                                    {"type": "state_change", "variable": variable, 
                                     "old_value": old_value, "new_value": value})
            elif self.active_logic == "b":
                self.memory_pool_b.add(f"Variable {variable} changed from {old_value} to {value}",
                                    {"type": "state_change", "variable": variable, 
                                     "old_value": old_value, "new_value": value})
            else:
                self.memory_pool_neutral.add(f"Variable {variable} changed from {old_value} to {value}",
                                          {"type": "state_change", "variable": variable, 
                                           "old_value": old_value, "new_value": value})
    
    def evaluate_expression(self, expression: str) -> float:
        """
        Evaluate a logical expression using the current variable states and active logic system.
        
        Args:
            expression: Logical expression to evaluate
            
        Returns:
            Float result of the expression (0.0, 0.5, or 1.0)
        """
        if self.active_logic == "a":
            result = self.logic_a.evaluate_expression(expression, self.variable_states)
            self.memory_pool_a.add(f"Evaluated with A: {expression} = {result}",
                                {"type": "evaluation", "expression": expression, "result": result})
            return result
        elif self.active_logic == "b":
            result = self.logic_b.evaluate_expression(expression, self.variable_states)
            self.memory_pool_b.add(f"Evaluated with B: {expression} = {result}",
                                {"type": "evaluation", "expression": expression, "result": result})
            return result
        else:
            # Neutral mode uses logic A's evaluation
            result = self.logic_a.evaluate_expression(expression, self.variable_states)
            self.memory_pool_neutral.add(f"Evaluated with neutral: {expression} = {result}",
                                      {"type": "evaluation", "expression": expression, "result": result})
            return result
    
    def apply_rules(self) -> Dict[str, float]:
        """
        Apply rules based on the active logic system.
        
        Returns:
            Dictionary of updated variable states
        """
        updated_states = self.variable_states.copy()
        
        if self.active_logic == "a":
            # Apply logic A rules (3-valued deductive reasoning)
            rules = sorted(self.logic_a.rules, key=lambda r: r["priority"], reverse=True)
            
            for rule in rules:
                certainty = self.logic_a.evaluate_expression(rule['condition'], self.variable_states)
                
                # Apply rule if certainty is at least 0.5 (unknown or true)
                if certainty >= 0.5:
                    # Parse the consequence
                    consequence = rule['consequence']
                    if '=' in consequence:
                        var, value_expr = consequence.split('=')
                        var = var.strip()
                        
                        # Determine the value based on the expression and certainty
                        if value_expr.strip() == "True":
                            value = certainty
                        elif value_expr.strip() == "False":
                            value = kleene_not(certainty)
                        elif value_expr.strip() == "Unknown":
                            value = 0.5
                        else:
                            try:
                                value = float(value_expr.strip())
                                # Normalize to 3-valued logic
                                if value < 0.25:
                                    value = 0.0
                                elif value < 0.75:
                                    value = 0.5
                                else:
                                    value = 1.0
                            except:
                                value = 0.5  # Default to unknown
                                
                        if var in updated_states:
                            updated_states[var] = value
                            self.memory_pool_a.add(f"Rule applied (A): {rule['condition']} → {var}={value} (certainty: {certainty})",
                                               {"type": "inference", "rule": rule, "result": {var: value}, "certainty": certainty})
                            self.inference_history.append({
                                "logic": "a",
                                "rule": rule,
                                "variables_before": self.variable_states.copy(),
                                "variables_after": updated_states.copy(),
                                "certainty": certainty,
                                "timestamp": len(self.inference_history)
                            })
                            
        elif self.active_logic == "b":
            # Apply logic B rules (3-valued inductive reasoning)
            rules = sorted(self.logic_b.rules, key=lambda r: r["priority"], reverse=True)
            
            for rule in rules:
                certainty = self.logic_b.evaluate_expression(rule['condition'], self.variable_states)
                
                # Only apply if certainty exceeds threshold
                if certainty >= self.logic_b.certainty_threshold:
                    # Parse the consequence
                    consequence = rule['consequence']
                    if '=' in consequence:
                        var, value_expr = consequence.split('=')
                        var = var.strip()
                        
                        # In 3-valued logic, we propagate certainty
                        if value_expr.strip() == "True":
                            value = certainty
                        elif value_expr.strip() == "False":
                            value = lukasiewicz_not(certainty)
                        elif value_expr.strip() == "Unknown":
                            value = 0.5
                        else:
                            try:
                                value = float(value_expr.strip())
                                # Normalize to 3-valued logic
                                if value < 0.25:
                                    value = 0.0
                                elif value < 0.75:
                                    value = 0.5
                                else:
                                    value = 1.0
                            except:
                                value = 0.5  # Default to unknown
                                
                        if var in updated_states:
                            # In 3-valued logic, we might combine certainties
                            current_value = updated_states[var]
                            
                            # Use Lukasiewicz operators to combine values
                            if current_value == 0.5:  # If current is unknown, use new value
                                updated_states[var] = value
                            elif value == 0.5:  # If new is unknown, keep current
                                pass
                            else:  # Otherwise combine using OR (for disjunctive combination)
                                updated_states[var] = lukasiewicz_or(current_value, value)
                                
                            self.memory_pool_b.add(
                                f"Rule applied (B): {rule['condition']} → {var}={value} (certainty: {certainty})",
                                {"type": "inference", "rule": rule, "result": {var: value}, "certainty": certainty}
                            )
        return updated_states


# ------------------------------------------------------
# Main function with MNIST Training Code
# ------------------------------------------------------
def main():
    """
    Main function to run the argumentation framework, logic behavior system,
    and then train a simple MNIST classifier for 10 epochs.
    """
    # --- Demo of logic systems ---
    # Create an argumentation framework with some arguments and attacks
    args = ["a", "b", "c", "d"]
    attacks = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
    af = ArgumentationFramework(args, attacks)
    
    # Create logic behaviors
    logic_a = LogicBehavior("Behavior A", "Conservative reasoning", "kleene")
    logic_a.add_rule("a and not b", "c=True", 2)
    logic_a.add_rule("b and c", "d=False", 1)
    
    logic_b = LogicBehavior("Behavior B", "Aggressive reasoning", "lukasiewicz")
    logic_b.add_rule("a or b", "c=True", 1)
    logic_b.add_rule("not c", "d=True", 2)
    
    # Initialize memory pools
    memory_a = MemoryPool(capacity=100, pool_type="a")
    memory_b = MemoryPool(capacity=100, pool_type="b")
    
    # Set up initial variable states
    initial_states = {"a": 1.0, "b": 0.5, "c": 0.0, "d": 0.5}
    
    # Run the system for a few iterations (demo)
    for i in range(5):
        print(f"Iteration {i+1}")
        # (In a full system you might call apply_rules here)
        print(f"Current states: {initial_states}")
        print("Memory A size:", len(memory_a))
        print("Memory B size:", len(memory_b))
        print("-" * 40)
    
    # Generate and print encodings
    print("Elementary encoding:", af.elementary_encoding())
    print("Normal encoding:", af.normal_encoding())
    
    print("Completed logic system demo.")
    
    # --- MNIST Training Code ---
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define a simple neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Initialize model, optimizer and loss function
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())
    
    # Training loop for 10 epochs
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/10 [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Evaluate on test set after each epoch
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Epoch: {epoch+1}/10, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    print("MNIST training completed.")

# If this file is run directly, execute the main function
if __name__ == "__main__":
    main()

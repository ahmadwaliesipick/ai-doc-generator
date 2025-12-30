import tree_sitter
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser

def extract_functions(file_path):
    # Initialize Parser
    JS_LANGUAGE = Language(tsjs.language())
    parser = Parser(JS_LANGUAGE)

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    # Query for function declarations and their comments
    # We look for comments that are siblings of the function node and appear right before it
    
    results = []
    
    # Using a simple recursive walk for now, can be optimized with queries
    def walk(node):
        if node.type in ['function_declaration', 'method_definition']:
            # Find preceding comment
            comment = ""
            curr = node.prev_sibling
            while curr:
                if curr.type == 'comment':
                    comment = code[curr.start_byte:curr.end_byte]
                    break
                elif curr.type in ['\n', ' ']: # Check for whitespace-only nodes if they exist
                    curr = curr.prev_sibling
                else:
                    # If we hit something else, stop looking for comments
                    break
            
            # Get function content
            function_code = code[node.start_byte:node.end_byte]
            
            if comment: # Only keep if it has a comment
                results.append({
                    'comment': comment,
                    'code': function_code,
                    'line': node.start_point[0]
                })
        
        for child in node.children:
            walk(child)

    walk(root_node)
    return results

if __name__ == "__main__":
    # Test with a dummy string if no file provided
    test_code = """
    /**
     * Adds two numbers.
     */
    function add(a, b) {
        return a + b;
    }

    // This subtracts b from a
    function subtract(a, b) {
        return a - b;
    }

    class Math {
        /**
         * Multiplies two numbers.
         */
        multiply(a, b) {
            return a * b;
        }
    }
    """
    import os
    test_file = "test_js.js"
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    extracted = extract_functions(test_file)
    for entry in extracted:
        print(f"--- Function at line {entry['line']} ---")
        print(f"Comment: {entry['comment']}")
        print(f"Code: {entry['code']}\n")
    
    os.remove(test_file)

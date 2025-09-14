#!/bin/bash
# Simple URDF to SDF conversion script using Gazebo's built-in converter

usage() {
    echo "Usage: $0 <input.urdf> [output.sdf]"
    echo "Converts URDF file to SDF format"
    echo ""
    echo "Arguments:"
    echo "  input.urdf   Input URDF file"
    echo "  output.sdf   Output SDF file (optional, defaults to input_name.sdf)"
    echo ""
    echo "Examples:"
    echo "  $0 robot.urdf"
    echo "  $0 robot.urdf my_robot.sdf"
    exit 1
}

# Check arguments
if [ $# -lt 1 ]; then
    usage
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Generate output filename if not provided
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="${INPUT_FILE%.*}.sdf"
fi

echo "Converting URDF to SDF..."
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

# Method 1: Try using gz sdf (Gazebo's built-in converter)
if command -v gz > /dev/null 2>&1; then
    echo "Using Gazebo's built-in converter..."
    gz sdf -p "$INPUT_FILE" > "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Conversion successful!"
        
        # Validate the output
        echo "Validating output SDF..."
        if gz sdf -k "$OUTPUT_FILE" > /dev/null 2>&1; then
            echo "✓ SDF validation passed!"
        else
            echo "⚠ Warning: SDF validation failed, but file was generated"
        fi
        
        echo "SDF file created: $OUTPUT_FILE"
        
        # Show file info
        echo ""
        echo "File information:"
        echo "  Size: $(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null) bytes"
        echo "  Lines: $(wc -l < "$OUTPUT_FILE") lines"
        
    else
        echo "✗ Conversion failed using gz sdf"
        exit 1
    fi

# Method 2: Try using rosrun if ROS is available
elif command -v rosrun > /dev/null 2>&1; then
    echo "Gazebo 'gz' command not found, trying ROS xacro..."
    if rosrun xacro xacro "$INPUT_FILE" > temp_processed.urdf 2>/dev/null; then
        echo "URDF processed successfully"
        
        # Try to find a Gazebo installation for conversion
        if [ -f "/usr/share/gazebo/setup.sh" ]; then
            source /usr/share/gazebo/setup.sh
            gzsdf print temp_processed.urdf > "$OUTPUT_FILE" 2>/dev/null
            rm -f temp_processed.urdf
        else
            echo "Warning: Gazebo not found, copying processed URDF as SDF"
            mv temp_processed.urdf "$OUTPUT_FILE"
        fi
        echo "✓ File created: $OUTPUT_FILE"
    else
        echo "✗ ROS xacro processing failed"
        exit 1
    fi

# Method 3: Basic text replacement (fallback)
else
    echo "Neither 'gz' nor 'rosrun' found, using basic conversion..."
    echo "⚠ Warning: This is a basic text replacement, may not handle all cases properly"
    
    # Basic conversion by replacing robot tag with sdf structure
    sed 's/<robot name="\([^"]*\)"/<sdf version="1.6"><model name="\1"/g; s/<\/robot>/<\/model><\/sdf>/g' "$INPUT_FILE" > "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Basic conversion completed: $OUTPUT_FILE"
        echo "⚠ Note: Manual verification recommended for complex models"
    else
        echo "✗ Basic conversion failed"
        exit 1
    fi
fi

# Additional helpful information
echo ""
echo "Next steps:"
echo "1. Verify the SDF file structure"
echo "2. Test spawning in Gazebo:"
echo "   gz service -s /world/YOUR_WORLD/create \\"
echo "     --reqtype gz.msgs.EntityFactory \\"
echo "     --reptype gz.msgs.Boolean \\"
echo "     --req \"sdf_filename: \\\"$(realpath "$OUTPUT_FILE")\\\"\""
echo ""
echo "3. For ROS integration, use the original URDF with robot_state_publisher"
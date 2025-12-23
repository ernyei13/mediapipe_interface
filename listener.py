import mido

print("Searching for MIDI input...")

try:
    # Look for the port created by the main app
    with mido.open_input('HandInterfacePort') as inport:
        print(f"Connected to: {inport.name}")
        print("Listening for signals (Ctrl+C to stop)...\n")
        
        for msg in inport:
            if msg.type == 'control_change':
                print(f"🎛️ CONTROL: CC#{msg.control} | Value: {msg.value}")
            elif msg.type == 'note_on':
                print(f"🟢 BUTTON ON:  Note {msg.note}")
            elif msg.type == 'note_off':
                print(f"🔴 BUTTON OFF: Note {msg.note}")
                
except IOError:
    print("Error: Could not find 'HandInterfacePort'. Make sure the main app is running!")
except KeyboardInterrupt:
    print("\nListener stopped.")
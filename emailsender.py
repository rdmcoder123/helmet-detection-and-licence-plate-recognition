import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tkinter as tk
from tkinter import filedialog, messagebox

def send_email():
    fromaddr = "swamynathantvm62@gmail.com"  # Default sender email
    toaddr = receiver_email_entry.get()  # Receiver email from GUI
    license_plate = license_plate_entry.get()  # License plate from GUI
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Violation Notice"
    
    # Body of the email
    body = f"Dear Sir/Madam,\n\nThis is to inform you about the violation related to license plate: {license_plate}.\n\nPlease find the details below."
    if attach_file_var.get():
        body += f"\n\nAttached file for details."
    if drive_link_var.get():
        body += f"\n\nOr view the video using the following link: {drive_link_entry.get()}"
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach the file if selected
    if attach_file_var.get():
        file_path = filedialog.askopenfilename(title="Select file to attach")
        if file_path:
            try:
                with open(file_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename="{file_path.split("/")[-1]}"')
                    msg.attach(part)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to attach file: {e}")
                return

    # Connect to Gmail SMTP server
    try:
        mailServer = smtplib.SMTP("smtp.gmail.com", 587)
        mailServer.starttls()
        mailServer.login(fromaddr, "vdcx qbgd nncm pxoz")  # Your app password here
        mailServer.sendmail(fromaddr, toaddr, msg.as_string())
        mailServer.quit()
        messagebox.showinfo("Success", "Email sent successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send email: {e}")

# Create the GUI
root = tk.Tk()
root.title("Violation Challan Sender")

tk.Label(root, text="Receiver Email:").pack(pady=5)
receiver_email_entry = tk.Entry(root, width=50)
receiver_email_entry.pack(pady=5)

tk.Label(root, text="License Plate Details:").pack(pady=5)
license_plate_entry = tk.Entry(root, width=50)
license_plate_entry.pack(pady=5)

attach_file_var = tk.BooleanVar()
attach_file_checkbox = tk.Checkbutton(root, text="Attach file", variable=attach_file_var)
attach_file_checkbox.pack(pady=5)

drive_link_var = tk.BooleanVar()
drive_link_checkbox = tk.Checkbutton(root, text="Include Google Drive link", variable=drive_link_var)
drive_link_checkbox.pack(pady=5)

drive_link_entry = tk.Entry(root, width=50)
drive_link_entry.pack(pady=5)
drive_link_entry.config(state=tk.DISABLED)  # Disabled by default

def toggle_drive_link_entry(*args):
    if drive_link_var.get():
        drive_link_entry.config(state=tk.NORMAL)
    else:
        drive_link_entry.config(state=tk.DISABLED)

drive_link_var.trace_add("write", toggle_drive_link_entry)

tk.Button(root, text="Send Email", command=send_email).pack(pady=20)

root.mainloop()

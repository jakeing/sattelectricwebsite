
"""]
RMJ Work Order Management System
A Flask application for managing work orders, time tracking, and project management.
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================
import os
from datetime import datetime, timedelta, time as time_class, date
import time
from io import BytesIO
from functools import wraps
import json

# Flask and extensions
from flask import (
    Flask, render_template, request, redirect, url_for, send_from_directory,
    send_file, session, jsonify, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_migrate import Migrate
import pandas as pd

# Utilities
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from functools import wraps
import time
from collections import defaultdict


# =============================================================================
# APP CONFIGURATION
# =============================================================================
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///workorders.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Security configuration
app.config['SECRET_KEY'] = 'mysecret'
app.config['DELETE_PASSWORD'] = 'secret123'  # Password required for deleting work orders

# File uploads configuration
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'rmj.dashboard@gmail.com'
app.config['MAIL_PASSWORD'] = 'ypwl msgw bwoq qhuk'
app.config['MAIL_DEFAULT_SENDER'] = 'rmj.dashboard@gmail.com'

# Notification configuration
app.config['REPORT_NOTIFICATION_ENABLED'] = True
app.config['REPORT_NOTIFICATION_EMAIL'] = 'beverlyn@rmj-consulting.com'
app.config['REPORT_NOTIFICATION_KEYWORDS'] = ['report', 'assessment']

# Initialize extensions
db = SQLAlchemy(app)
mail = Mail(app)
migrate = Migrate(app, db)


# =============================================================================
# DATABASE MODELS
# =============================================================================
class User(db.Model):
    """User model for authentication and access control"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    full_name = db.Column(db.String(100), nullable=True)
    worker_role_id = db.Column(db.Integer, db.ForeignKey('worker_role.id'), nullable=True)  # COMMENT THIS OUT TEMPORARILY
    
    def set_password(self, password):
        """Set the password hash from plain text password"""
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        """Check if password matches the hash"""
        return check_password_hash(self.password_hash, password)
    
class WorkerRole(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    hourly_rate = db.Column(db.Float, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    assigned_users = db.relationship('User', backref='worker_role', lazy=True)  # COMMENT THIS OUT TEMPORARILY


class WorkOrder(db.Model):
    """Work order model to track jobs and assignments"""
    id = db.Column(db.Integer, primary_key=True)
    customer_work_order_number = db.Column(db.String(50), nullable=True)
    rmj_job_number = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(50))
    owner = db.Column(db.String(50))
    estimated_hours = db.Column(db.Float, nullable=True, default=0)
    priority = db.Column(db.String(20))
    location = db.Column(db.String(100))
    scheduled_date = db.Column(db.Date)
    requested_by = db.Column(db.String(80), nullable=True)
    classification = db.Column(db.String(50), default='Billable')
    approved_for_work = db.Column(db.Boolean, default=False, nullable=False)
    project_cost = db.Column(db.Float, nullable=True, default=0.0)
    # Relationships
    time_entries = db.relationship('TimeEntry', backref='work_order', lazy=True, cascade="all, delete")
    documents = db.relationship('WorkOrderDocument', backref='work_order', lazy=True, cascade="all, delete")

    @property
    def hours_logged(self):
        """Calculate total hours logged for this work order"""
        return sum(entry.hours_worked for entry in self.time_entries)

    @property
    def hours_remaining(self):
        """Calculate remaining hours based on estimate"""
        try:
            estimated = float(self.estimated_hours)
        except (TypeError, ValueError):
            estimated = 0.0
        return estimated - self.hours_logged

    @property
    def has_report(self):
        """Check if this work order has any report documents"""
        for doc in self.documents:
             if doc.document_type == 'report' or "report" in doc.original_filename.lower():
                return True
        return False
    
    @property
    def has_approved_report(self):
        """Check if this work order has any approved report documents"""
        for doc in self.documents:
            if "report" in doc.original_filename.lower() and doc.is_approved:
                return True
        return False

class WorkOrderSignature(db.Model):
    """Digital signatures for work orders"""
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey('work_order.id'), nullable=False)
    signee_name = db.Column(db.String(100), nullable=False)
    signature_data = db.Column(db.Text, nullable=False)  # Base64 encoded signature image
    signed_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)
    
    # Relationships
    work_order = db.relationship('WorkOrder', backref='signatures', foreign_keys=[work_order_id])

class TimeEntry(db.Model):
    """Time tracking entries for work orders"""
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey('work_order.id'), nullable=False)
    task_id = db.Column(db.Integer, db.ForeignKey('project_task.id', name='fk_time_entry_task'), nullable=True)
    engineer = db.Column(db.String(50), nullable=False)
    work_date = db.Column(db.Date, nullable=False)
    time_in = db.Column(db.Time, nullable=False)
    time_out = db.Column(db.Time, nullable=False)
    hours_worked = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    entered_on_jl = db.Column(db.Boolean, default=False)
    entered_on_jt = db.Column(db.Boolean, default=False)
    
    # ADD THESE NEW FIELDS:
    role_at_time_of_entry = db.Column(db.String(100), nullable=True)  # Store role name when time was logged
    rate_at_time_of_entry = db.Column(db.Float, nullable=True)        # Store hourly rate when time was logged
    
    # Relationships
    task = db.relationship('ProjectTask', backref='time_entries', foreign_keys=[task_id])


class WorkOrderDocument(db.Model):
    """Document attachments for work orders"""
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey('work_order.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True)
    filename = db.Column(db.String(100), nullable=False)
    original_filename = db.Column(db.String(100))
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    is_approved = db.Column(db.Boolean, default=False)
    document_type = db.Column(db.String(50), default='regular')
    
    # Relationships
    project = db.relationship('Project', backref='documents', foreign_keys=[project_id])

class ChangeLog(db.Model):
    """Audit log for tracking all changes in the system"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    user_id = db.Column(db.Integer, nullable=True)  # The user who performed the action
    action = db.Column(db.String(255), nullable=False)  # e.g. "Created WorkOrder"
    object_type = db.Column(db.String(50), nullable=False)  # e.g. "WorkOrder", "TimeEntry"
    object_id = db.Column(db.Integer, nullable=True)  # The id of the affected object
    description = db.Column(db.Text, nullable=True)  # More detailed info


class Project(db.Model):
    """Project management for work orders"""
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey('work_order.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    status = db.Column(db.String(50), default='Planning')  # Planning, In Progress, Completed, On Hold
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    tasks = db.relationship('ProjectTask', backref='project', lazy=True, cascade="all, delete")
    work_order = db.relationship('WorkOrder', backref='project', uselist=False)


class ProjectTask(db.Model):
    """Tasks within a project"""
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    status = db.Column(db.String(50), default='Not Started')  # Not Started, In Progress, Completed, Delayed
    estimated_hours = db.Column(db.Float, default=0)
    actual_hours = db.Column(db.Float, default=0)
    priority = db.Column(db.String(20), default='Medium')  # Low, Medium, High
    dependencies = db.Column(db.String(200))  # Comma-separated list of task IDs this task depends on
    assigned_to = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    position = db.Column(db.Integer, default=0)
    progress_percent = db.Column(db.Integer, nullable=True)  # Manually set progress (0-100)
    
    @property
    def hours_remaining(self):
        """Calculate remaining hours for this task"""
        return self.estimated_hours - self.actual_hours
        
    @property
    def completion_percentage(self):
        """Calculate the task completion percentage"""
        if self.status == 'Completed':
            return 100
        elif self.progress_percent is not None:
            return self.progress_percent  # Return manually set progress if available
        elif self.estimated_hours == 0:
            return 0
        else:
            percentage = (self.actual_hours / self.estimated_hours) * 100
            return min(int(percentage), 99)  # Cap at 99% until marked complete
    
    @property
    def dependent_tasks(self):
        """Get list of tasks this task depends on"""
        if not self.dependencies:
            return []
        task_ids = [int(id.strip()) for id in self.dependencies.split(',') if id.strip().isdigit()]
        return ProjectTask.query.filter(ProjectTask.id.in_(task_ids)).all()
    
    @property
    def actual_hours_from_entries(self):
        """Calculate actual hours from linked time entries"""
        return sum(entry.hours_worked for entry in self.time_entries)
        
    def update_actual_hours(self):
        """Update actual_hours based on linked time entries"""
        self.actual_hours = self.actual_hours_from_entries
        return self.actual_hours

# Add these to the DATABASE MODELS section in app.py

class NotificationSetting(db.Model):
    """Settings for email notifications"""
    id = db.Column(db.Integer, primary_key=True)
    notification_type = db.Column(db.String(50), nullable=False, unique=True)
    enabled = db.Column(db.Boolean, default=False)
    options = db.Column(db.JSON, default={})  # Store type-specific settings
    
    # Relationships
    recipients = db.relationship('NotificationRecipient', backref='notification_setting', 
                                cascade="all, delete-orphan")

class NotificationRecipient(db.Model):
    """Recipients for different notification types"""
    id = db.Column(db.Integer, primary_key=True)
    notification_setting_id = db.Column(db.Integer, db.ForeignKey('notification_setting.id'), 
                                      nullable=False)
    email = db.Column(db.String(100), nullable=False)


# =============================================================================
# AUTHENTICATION DECORATORS
# =============================================================================
def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
         if 'user_id' not in session:
             return redirect(url_for('login'))
         return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to require admin privileges for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        print(f"User check: {user.username}, Role: {user.role}")  # Debug logging
        if not user or (user.role.lower() != 'admin' and user.username.lower() != 'admin'):
            return "Access denied", 403
        return f(*args, **kwargs)
    return decorated_function

# Rate limiting decorator
def rate_limit(max_calls=10, time_window=60):
    """Rate limit decorator to prevent too many requests"""
    def decorator(f):
        # Store request counts per user
        request_counts = defaultdict(list)
        
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': 'Authentication required'}), 401
            
            user_id = session['user_id']
            now = time.time()
            
            # Clean up old requests outside the time window
            request_counts[user_id] = [
                req_time for req_time in request_counts[user_id] 
                if now - req_time < time_window
            ]
            
            # Check if rate limit exceeded
            if len(request_counts[user_id]) >= max_calls:
                return jsonify({
                    'success': False, 
                    'message': f'Rate limit exceeded. Maximum {max_calls} requests per {time_window} seconds.'
                }), 429
            
            # Record this request
            request_counts[user_id].append(now)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def populate_user_full_names():
    """Populate full_name for existing users based on mapping"""
    # Keep the existing mapping for migration
    USER_MAPPING = {
        "CHinkle": "Curtis Hinkle",
        "RHinkle": "Ron Hinkle",
        "ASeymour": "Andrew Seymour",
        "AAviles": "Alex Aviles",
        "AYork": "Austin York",
        "MWestmoreland": "Micky Westmoreland",
        "BNewton": "Beverly Newton",
        "BParis": "Benjamin Paris"
    }
    
    # Update existing users
    for username, full_name in USER_MAPPING.items():
        user = User.query.filter_by(username=username).first()
        if user and not user.full_name:
            user.full_name = full_name
    
    db.session.commit()
    print("Updated full names for existing users")

def get_user_role_and_rate(engineer_name):
    """Get the current role and rate for an engineer by name"""
    # Find user by full name or username
    user = User.query.filter(
        (User.full_name == engineer_name) | (User.username == engineer_name)
    ).first()
    
    if not user:
        return None, 0.0
    
    # Check if user has worker_role_id field and it's assigned
    if hasattr(user, 'worker_role_id') and user.worker_role_id:
        worker_role = WorkerRole.query.get(user.worker_role_id)
        if worker_role:
            return worker_role.name, worker_role.hourly_rate
    
    return None, 0.0

def add_worker_role_id_column():
    """Add worker_role_id column to User table if it doesn't exist"""
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        
        # Check if user table exists
        tables = inspector.get_table_names()
        if 'user' not in tables:
            print("User table doesn't exist yet")
            return False
        
        # Check if worker_role_id column already exists
        columns = [col['name'] for col in inspector.get_columns('user')]
        
        if 'worker_role_id' in columns:
            print("worker_role_id column already exists")
            return True
        
        # Add the column
        with db.engine.connect() as conn:
            conn.execute(text('ALTER TABLE user ADD COLUMN worker_role_id INTEGER'))
            conn.commit()
        
        print("Added worker_role_id column to User table")
        return True
        
    except Exception as e:
        print(f"Error adding worker_role_id column: {e}")
        return False

def get_engineer_name(username):
    """Get the full name of an engineer from their username"""
    user = User.query.filter_by(username=username).first()
    if user and user.full_name:
        return user.full_name
    # If no mapping found, just return the username as is
    return username


def parse_date(date_str):
    """Parse a date string into a date object"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return None


def parse_time(time_str):
    """Parse a time string into a time object"""
    try:
        return datetime.strptime(time_str, '%H:%M').time()
    except (ValueError, TypeError):
        return None


def calculate_hours(date_obj, time_in, time_out):
    """Calculate hours worked between time_in and time_out"""
    dt_time_in = datetime.combine(date_obj, time_in)
    dt_time_out = datetime.combine(date_obj, time_out)
    if dt_time_out < dt_time_in:
        dt_time_out += timedelta(days=1)
    return (dt_time_out - dt_time_in).total_seconds() / 3600.0


def get_week_dates(year, week):
    """Get the start and end dates for a week"""
    try:
        # Calculate the first Sunday of the week
        start_date = (datetime.fromisocalendar(year, week, 1) - timedelta(days=1)).date()
        end_date = start_date + timedelta(days=6)
        return start_date, end_date
    except Exception:
        return None, None


def log_change(user_id, action, object_type, object_id=None, description=""):
    """Log a change to the database"""
    log_entry = ChangeLog(
        user_id=user_id,
        action=action,
        object_type=object_type,
        object_id=object_id,
        description=description
    )
    db.session.add(log_entry)

def migrate_existing_time_entries():
    """One-time migration to populate role/rate for existing time entries"""
    try:
        entries_to_update = TimeEntry.query.filter(
            TimeEntry.role_at_time_of_entry.is_(None)
        ).all()
        
        print(f"Found {len(entries_to_update)} time entries to migrate...")
        
        for entry in entries_to_update:
            current_role, current_rate = get_user_role_and_rate(entry.engineer)
            entry.role_at_time_of_entry = current_role or 'Legacy Entry'
            entry.rate_at_time_of_entry = current_rate or 0.0
        
        db.session.commit()
        print(f"Successfully migrated {len(entries_to_update)} time entries")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        db.session.rollback()

def add_project_cost_column():
    """Add project_cost column if it doesn't exist"""
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('work_order')]
        
        if 'project_cost' not in columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE work_order ADD COLUMN project_cost FLOAT DEFAULT 0.0'))
                conn.commit()
            print("Added project_cost column")
        return True
    except Exception as e:
        print(f"Error adding project_cost column: {e}")
        return False


def get_sorted_work_orders(status, sort_by='id', order='asc'):
    """Get work orders with the given status, sorted by the given column"""
    valid_sort_columns = {
        'id': WorkOrder.id,
        'customer_work_order_number': WorkOrder.customer_work_order_number,
        'rmj_job_number': WorkOrder.rmj_job_number,
        'description': WorkOrder.description,
        'status': WorkOrder.status,
        'owner': WorkOrder.owner,
        'estimated_hours': WorkOrder.estimated_hours,
        'priority': WorkOrder.priority,
        'location': WorkOrder.location,
        'scheduled_date': WorkOrder.scheduled_date,
        'approved_for_work': WorkOrder.approved_for_work
    }
    sort_column = valid_sort_columns.get(sort_by, WorkOrder.id)
    sort_column = sort_column.desc() if order == 'desc' else sort_column.asc()
    return WorkOrder.query.filter_by(status=status).order_by(sort_column).all()


def is_report_file(filename):
    """Check if a filename contains any of the report keywords."""
    lower_filename = filename.lower()
    return any(keyword in lower_filename for keyword in app.config['REPORT_NOTIFICATION_KEYWORDS'])

def add_worker_role_id_column():
    """Add worker_role_id column to User table if it doesn't exist"""
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        
        # Check if user table exists
        tables = inspector.get_table_names()
        if 'user' not in tables:
            print("User table doesn't exist yet")
            return False
        
        # Check if worker_role_id column already exists
        columns = [col['name'] for col in inspector.get_columns('user')]
        
        if 'worker_role_id' in columns:
            print("worker_role_id column already exists")
            return True
        
        # Add the column
        with db.engine.connect() as conn:
            conn.execute(text('ALTER TABLE user ADD COLUMN worker_role_id INTEGER'))
            conn.commit()
        
        print("Added worker_role_id column to User table")
        return True
        
    except Exception as e:
        print(f"Error adding worker_role_id column: {e}")
        return False
    
def add_time_entry_role_columns():
    """Add role_at_time_of_entry and rate_at_time_of_entry columns to TimeEntry table if they don't exist"""
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        
        # Check if time_entry table exists
        tables = inspector.get_table_names()
        if 'time_entry' not in tables:
            print("time_entry table doesn't exist yet")
            return False
        
        # Check if columns already exist
        columns = [col['name'] for col in inspector.get_columns('time_entry')]
        
        columns_to_add = []
        if 'role_at_time_of_entry' not in columns:
            columns_to_add.append('role_at_time_of_entry')
        if 'rate_at_time_of_entry' not in columns:
            columns_to_add.append('rate_at_time_of_entry')
        
        if not columns_to_add:
            print("TimeEntry role/rate columns already exist")
            return True
        
        # Add the columns
        with db.engine.connect() as conn:
            for column in columns_to_add:
                if column == 'role_at_time_of_entry':
                    conn.execute(text('ALTER TABLE time_entry ADD COLUMN role_at_time_of_entry VARCHAR(100)'))
                    print("Added role_at_time_of_entry column to TimeEntry table")
                elif column == 'rate_at_time_of_entry':
                    conn.execute(text('ALTER TABLE time_entry ADD COLUMN rate_at_time_of_entry FLOAT'))
                    print("Added rate_at_time_of_entry column to TimeEntry table")
            conn.commit()
        
        print("Successfully added TimeEntry role/rate columns")
        return True
        
    except Exception as e:
        print(f"Error adding TimeEntry role/rate columns: {e}")
        return False

def send_report_notification(work_order, document):
    """Send an email notification when a report is uploaded."""
    try:
        recipient = app.config['REPORT_NOTIFICATION_EMAIL']
        subject = f"Report Uploaded for Work Order {work_order.rmj_job_number}"
        
        body = f"""
        A new report has been uploaded for Work Order:
        
        RMJ Job Number: {work_order.rmj_job_number}
        Customer Work Order Number: {work_order.customer_work_order_number}
        Description: {work_order.description}
        
        Document: {document.original_filename}
        Uploaded at: {document.upload_time.strftime('%Y-%m-%d %H:%M:%S')}
        
        You can view the work order details at: {url_for('work_order_detail', work_order_id=work_order.id, _external=True)}
        """
        
        msg = Message(subject=subject, recipients=[recipient], body=body, sender=app.config['MAIL_DEFAULT_SENDER'])
        msg.extra_headers = {
            'X-Priority': '1',
            'X-MSMail-Priority': 'High',
            'Importance': 'High',
            'X-Auto-Response-Suppress': 'OOF, DR, RN, NRN, AutoReply'
        }
        mail.send(msg)
        
        # Log the notification
        log_change(None, "Sent Report Notification", "Email", None, 
                  f"Sent notification for report upload on Work Order #{work_order.id}")
        return True
    except Exception as e:
        # Log the error but don't crash the application
        print(f"Error sending email notification: {e}")
        log_change(None, "Failed Report Notification", "Error", None, 
                  f"Failed to send notification for report upload: {str(e)}")
        return False

# Add these to the HELPER FUNCTIONS section

def get_notification_setting(notification_type):
    """Get notification settings for a given type"""
    setting = NotificationSetting.query.filter_by(notification_type=notification_type).first()
    if setting and setting.enabled:
        return setting
    return None

def get_notification_recipients(setting, work_order=None):
    """Get list of recipients for a notification"""
    recipients = []
    
    # Add all configured recipients for this notification type
    if setting and setting.recipients:
        recipients = [r.email for r in setting.recipients]
    
    # If no recipients, use the default email
    if not recipients:
        default_email = app.config.get('REPORT_NOTIFICATION_EMAIL')
        if default_email:
            recipients.append(default_email)
    
    # Add work order owner if option is enabled and owner has email
    if work_order and setting:
        options = setting.options
        if (notification_type == 'hours_threshold' and options.get('include_work_order_owner')) or \
           (notification_type == 'scheduled_date' and options.get('include_owner')):
            owner = work_order.owner
            if owner and '@' in owner:  # Simple check if owner field contains an email
                if owner not in recipients:
                    recipients.append(owner)
    
    return recipients

def send_report_notification(work_order, document):
    """Send an email notification when a report is uploaded."""
    setting = get_notification_setting('report_upload')
    if not setting:
        return False
    
    try:
        recipients = get_notification_recipients(setting)
        if not recipients:
            return False
        
        subject = f"Report Uploaded for Work Order {work_order.rmj_job_number}"
        
        body = f"""
        A new report has been uploaded for Work Order:
        
        RMJ Job Number: {work_order.rmj_job_number}
        Customer Work Order Number: {work_order.customer_work_order_number}
        Description: {work_order.description}
        
        Document: {document.original_filename}
        Uploaded at: {document.upload_time.strftime('%Y-%m-%d %H:%M:%S')}
        
        You can view the work order details at: {url_for('work_order_detail', work_order_id=work_order.id, _external=True)}
        """
        
        msg = Message(subject=subject, recipients=recipients, body=body, sender=app.config['MAIL_DEFAULT_SENDER'])
        mail.send(msg)
        
        # Log the notification
        log_change(None, "Sent Report Notification", "Email", None, 
                  f"Sent notification for report upload on Work Order #{work_order.id}")
        return True
    except Exception as e:
        print(f"Error sending email notification: {e}")
        log_change(None, "Failed Report Notification", "Error", None, 
                  f"Failed to send notification for report upload: {str(e)}")
        return False

def send_report_approval_notification(work_order, document):
    """Send notification when a report needs approval"""
    setting = get_notification_setting('report_approval')
    if not setting:
        return False
    
    try:
        recipients = get_notification_recipients(setting)
        if not recipients:
            return False
        
        subject = f"Report Requires Approval - Work Order {work_order.rmj_job_number}"
        
        body = f"""
        A report has been uploaded that requires approval:
        
        RMJ Job Number: {work_order.rmj_job_number}
        Customer Work Order Number: {work_order.customer_work_order_number}
        Description: {work_order.description}
        
        Document: {document.original_filename}
        Uploaded at: {document.upload_time.strftime('%Y-%m-%d %H:%M:%S')}
        
        You can view and approve this report at: {url_for('work_order_detail', work_order_id=work_order.id, _external=True)}
        """
        
        msg = Message(subject=subject, recipients=recipients, body=body, sender=app.config['MAIL_DEFAULT_SENDER'])
        mail.send(msg)
        
        log_change(None, "Sent Report Approval Notification", "Email", None, 
                  f"Sent approval notification for report on Work Order #{work_order.id}")
        return True
    except Exception as e:
        print(f"Error sending approval notification: {e}")
        return False

def send_status_change_notification(work_order, old_status, new_status):
    """Send notification when a work order status changes"""
    setting = get_notification_setting('status_change')
    if not setting:
        return False
    
    # Check if this status change should trigger a notification
    options = setting.options
    should_notify = False
    
    if old_status == 'Open' and new_status == 'Complete' and options.get('open_to_complete'):
        should_notify = True
    elif old_status == 'Complete' and new_status == 'Closed' and options.get('complete_to_closed'):
        should_notify = True
    elif new_status == 'Open' and options.get('any_to_open'):
        should_notify = True
        
    if not should_notify:
        return False
    
    try:
        recipients = get_notification_recipients(setting)
        if not recipients:
            return False
        
        subject = f"Work Order Status Changed - {work_order.rmj_job_number}"
        
        body = f"""
        A work order status has been changed:
        
        RMJ Job Number: {work_order.rmj_job_number}
        Customer Work Order Number: {work_order.customer_work_order_number}
        Description: {work_order.description}
        
        Previous Status: {old_status}
        New Status: {new_status}
        
        You can view the work order at: {url_for('work_order_detail', work_order_id=work_order.id, _external=True)}
        """
        
        msg = Message(subject=subject, recipients=recipients, body=body, sender=app.config['MAIL_DEFAULT_SENDER'])
        mail.send(msg)
        
        log_change(None, "Sent Status Change Notification", "Email", None, 
                  f"Sent notification for status change on Work Order #{work_order.id}")
        return True
    except Exception as e:
        print(f"Error sending status change notification: {e}")
        return False

def send_hours_threshold_notification(work_order, hours_logged, estimated_hours, percentage):
    """Send notification when hours threshold is reached"""
    setting = get_notification_setting('hours_threshold')
    if not setting:
        return False
    
    options = setting.options
    warning_threshold = options.get('warning_threshold', 80)
    exceeded_alert = options.get('exceeded_alert', True)
    
    # Check if we should send a notification
    should_notify = False
    notification_type = ""
    
    if percentage >= warning_threshold and percentage < 100:
        should_notify = True
        notification_type = "Warning"
    elif percentage >= 100 and exceeded_alert:
        should_notify = True
        notification_type = "Exceeded"
        
    if not should_notify:
        return False
    
    try:
        recipients = get_notification_recipients(setting, work_order)
        if not recipients:
            return False
        
        subject = f"Hours {notification_type} - Work Order {work_order.rmj_job_number}"
        
        body = f"""
        A work order has reached {percentage:.1f}% of its estimated hours:
        
        RMJ Job Number: {work_order.rmj_job_number}
        Customer Work Order Number: {work_order.customer_work_order_number}
        Description: {work_order.description}
        
        Hours Logged: {hours_logged:.1f}
        Estimated Hours: {estimated_hours:.1f}
        Percentage: {percentage:.1f}%
        
        You can view the work order at: {url_for('work_order_detail', work_order_id=work_order.id, _external=True)}
        """
        
        msg = Message(subject=subject, recipients=recipients, body=body, sender=app.config['MAIL_DEFAULT_SENDER'])
        mail.send(msg)
        
        log_change(None, f"Sent Hours {notification_type} Notification", "Email", None, 
                  f"Sent hours threshold notification for Work Order #{work_order.id}")
        return True
    except Exception as e:
        print(f"Error sending hours threshold notification: {e}")
        return False

def send_scheduled_date_reminder(work_order):
    """Send reminder for upcoming scheduled date"""
    setting = get_notification_setting('scheduled_date')
    if not setting:
        return False
    
    options = setting.options
    days_before = options.get('days_before', 3)
    
    # Check if the scheduled date is coming up
    if not work_order.scheduled_date:
        return False
        
    days_until = (work_order.scheduled_date - datetime.now().date()).days
    
    if days_until != days_before:  # Only send exactly when we hit the threshold
        return False
    
    try:
        recipients = get_notification_recipients(setting, work_order)
        if not recipients:
            return False
        
        subject = f"Upcoming Work Order - {work_order.rmj_job_number}"
        
        body = f"""
        A work order is scheduled in {days_before} days:
        
        RMJ Job Number: {work_order.rmj_job_number}
        Customer Work Order Number: {work_order.customer_work_order_number}
        Description: {work_order.description}
        
        Scheduled Date: {work_order.scheduled_date.strftime('%Y-%m-%d')}
        
        You can view the work order at: {url_for('work_order_detail', work_order_id=work_order.id, _external=True)}
        """
        
        msg = Message(subject=subject, recipients=recipients, body=body, sender=app.config['MAIL_DEFAULT_SENDER'])
        mail.send(msg)
        
        log_change(None, "Sent Scheduled Date Reminder", "Email", None, 
                  f"Sent scheduled date reminder for Work Order #{work_order.id}")
        return True
    except Exception as e:
        print(f"Error sending scheduled date reminder: {e}")
        return False

def send_new_work_order_notification(work_order):
    """Send notification for new work order"""
    setting = get_notification_setting('new_work_order')
    if not setting:
        return False
    
    options = setting.options
    
    # Check if this priority level should trigger a notification
    should_notify = False
    if work_order.priority == 'High' and options.get('high_priority'):
        should_notify = True
    elif work_order.priority == 'Medium' and options.get('medium_priority'):
        should_notify = True
    elif work_order.priority == 'Low' and options.get('low_priority'):
        should_notify = True
        
    if not should_notify:
        return False
    
    try:
        recipients = get_notification_recipients(setting)
        if not recipients:
            return False
        
        subject = f"New Work Order Created - {work_order.rmj_job_number}"
        
        body = f"""
        A new work order has been created:
        
        RMJ Job Number: {work_order.rmj_job_number}
        Customer Work Order Number: {work_order.customer_work_order_number}
        Description: {work_order.description}
        
        Priority: {work_order.priority}
        Owner: {work_order.owner}
        Estimated Hours: {work_order.estimated_hours}
        
        You can view the work order at: {url_for('work_order_detail', work_order_id=work_order.id, _external=True)}
        """
        
        msg = Message(subject=subject, recipients=recipients, body=body, sender=app.config['MAIL_DEFAULT_SENDER'])
        mail.send(msg)
        
        log_change(None, "Sent New Work Order Notification", "Email", None, 
                  f"Sent notification for new Work Order #{work_order.id}")
        return True
    except Exception as e:
        print(f"Error sending new work order notification: {e}")
        return False
    
def process_work_order_entries(time_entries, user_rates):
    """Efficiently process time entries for a work order"""
    engineer_data = {}
    
    for entry in time_entries:
        engineer = entry.engineer
        if not engineer:
            continue
            
        if engineer not in engineer_data:
            engineer_data[engineer] = {
                'hours': 0,
                'cost': 0,
                'entries': []
            }
        
        # Use historical rate if available, otherwise current rate
        if entry.rate_at_time_of_entry is not None:
            entry_rate = entry.rate_at_time_of_entry
            entry_role = entry.role_at_time_of_entry or 'Unknown'
        else:
            entry_rate = user_rates.get(engineer, 0.0)
            entry_role = 'Current Rate'
        
        entry_cost = entry.hours_worked * entry_rate
        
        engineer_data[engineer]['hours'] += entry.hours_worked
        engineer_data[engineer]['cost'] += entry_cost
        
        # Add entry details (limit to essential data)
        engineer_data[engineer]['entries'].append({
            'id': entry.id,
            'date': entry.work_date,
            'hours': entry.hours_worked,
            'rate': entry_rate,
            'role': entry_role,
            'cost': entry_cost,
            'description': entry.description
        })
    
    return engineer_data

def get_accounting_summary_stats(status_filter):
    """Get summary statistics efficiently"""
    try:
        # Use database aggregation instead of loading all records
        if status_filter == 'all':
            total_work_orders = WorkOrder.query.count()
            open_count = WorkOrder.query.filter_by(status='Open').count()
            completed_count = WorkOrder.query.filter_by(status='Complete').count()
            closed_count = WorkOrder.query.filter_by(status='Closed').count()
        else:
            total_work_orders = WorkOrder.query.filter_by(status=status_filter).count()
            open_count = WorkOrder.query.filter_by(status='Open').count() if status_filter == 'all' else (total_work_orders if status_filter == 'Open' else 0)
            completed_count = WorkOrder.query.filter_by(status='Complete').count() if status_filter == 'all' else (total_work_orders if status_filter == 'Complete' else 0)
            closed_count = WorkOrder.query.filter_by(status='Closed').count() if status_filter == 'all' else (total_work_orders if status_filter == 'Closed' else 0)
        
        # For grand totals, we'll calculate on current page only to avoid performance issues
        # You can add a "Calculate All" button if needed for full statistics
        
        return {
            'total_work_orders': total_work_orders,
            'open_count': open_count,
            'completed_count': completed_count,
            'closed_count': closed_count,
            'grand_total_hours': 0,  # Will be calculated per page
            'grand_total_cost': 0    # Will be calculated per page
        }
    except Exception as e:
        print(f"Error calculating summary stats: {e}")
        return {
            'total_work_orders': 0,
            'open_count': 0,
            'completed_count': 0,
            'closed_count': 0,
            'grand_total_hours': 0,
            'grand_total_cost': 0
        }

def add_database_indexes():
    """Add database indexes to improve accounting page performance"""
    try:
        from sqlalchemy import text
        
        # Add indexes for frequently queried columns
        indexes_to_add = [
            # TimeEntry indexes
            "CREATE INDEX IF NOT EXISTS idx_time_entry_work_order_id ON time_entry(work_order_id);",
            "CREATE INDEX IF NOT EXISTS idx_time_entry_engineer ON time_entry(engineer);",
            "CREATE INDEX IF NOT EXISTS idx_time_entry_work_date ON time_entry(work_date);",
            
            # WorkOrder indexes
            "CREATE INDEX IF NOT EXISTS idx_work_order_status ON work_order(status);",
            "CREATE INDEX IF NOT EXISTS idx_work_order_rmj_job_number ON work_order(rmj_job_number);",
            "CREATE INDEX IF NOT EXISTS idx_work_order_customer_work_order_number ON work_order(customer_work_order_number);",
            
            # User indexes
            "CREATE INDEX IF NOT EXISTS idx_user_worker_role_id ON user(worker_role_id);",
            "CREATE INDEX IF NOT EXISTS idx_user_full_name ON user(full_name);",
            
            # WorkerRole indexes
            "CREATE INDEX IF NOT EXISTS idx_worker_role_is_active ON worker_role(is_active);"
        ]
        
        with db.engine.connect() as conn:
            for index_sql in indexes_to_add:
                try:
                    conn.execute(text(index_sql))
                    print(f"Added index: {index_sql}")
                except Exception as e:
                    print(f"Index might already exist: {e}")
            conn.commit()
            
        print("Database indexes added successfully")
        return True
        
    except Exception as e:
        print(f"Error adding database indexes: {e}")
        return False

def get_cached_user_rates():
    """Get user rates with caching to avoid repeated queries"""
    # You can implement Redis caching here if needed
    # For now, just optimize the query
    
    user_rates = {}
    
    # Single query to get all users with their roles
    users_with_roles = db.session.query(
        User.username,
        User.full_name,
        WorkerRole.hourly_rate
    ).outerjoin(
        WorkerRole, User.worker_role_id == WorkerRole.id
    ).filter(
        (WorkerRole.is_active == True) | (WorkerRole.id.is_(None))
    ).all()
    
    for username, full_name, rate in users_with_roles:
        engineer_name = full_name or username
        user_rates[engineer_name] = rate or 0.0
    
    return user_rates


# Add a function to check scheduled date reminders (to be run daily)
def check_scheduled_date_reminders():
    """Check for work orders with upcoming scheduled dates and send reminders"""
    setting = get_notification_setting('scheduled_date')
    if not setting:
        return
    
    options = setting.options
    days_before = options.get('days_before', 3)
    
    # Calculate the date to check for
    target_date = datetime.now().date() + timedelta(days=days_before)
    
    # Find work orders scheduled for the target date
    work_orders = WorkOrder.query.filter_by(scheduled_date=target_date).all()
    
    for work_order in work_orders:
        send_scheduled_date_reminder(work_order)

@app.route('/admin/send_test_email', methods=['POST'])
@login_required
@admin_required
def send_test_email():
    """Send a test email to verify notification settings"""
    try:
        recipient = request.form.get('recipient')
        subject = request.form.get('subject', 'RMJ Dashboard Test Email')
        
        if not recipient:
            return jsonify({'success': False, 'message': 'No recipient provided'}), 400
            
        body = f"""
        This is a test email from the RMJ Dashboard notification system.
        
        If you received this email, your notification settings are working correctly.
        
        Time sent: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        msg = Message(subject=subject, recipients=[recipient], body=body, sender=app.config['MAIL_DEFAULT_SENDER'])
        mail.send(msg)
        
        # Log the test email
        log_change(
            session.get('user_id'),
            "Sent Test Email",
            "Email",
            None,
            f"Sent test email to {recipient}"
        )
        
        return jsonify({'success': True, 'message': f'Test email sent to {recipient}'})
    except Exception as e:
        print(f"Error sending test email: {e}")
        return jsonify({'success': False, 'message': f'Error sending email: {str(e)}'}), 500

@app.route('/admin/verify_email_configuration')
@login_required
@admin_required
def verify_email_configuration():
    """Verify that the email configuration is valid"""
    try:
        # Check if required email settings are present
        mail_server = app.config.get('MAIL_SERVER')
        mail_port = app.config.get('MAIL_PORT')
        mail_username = app.config.get('MAIL_USERNAME')
        mail_password = app.config.get('MAIL_PASSWORD')
        
        if not all([mail_server, mail_port, mail_username, mail_password]):
            return jsonify({
                'success': False,
                'message': 'Missing email configuration settings',
                'settings': {
                    'MAIL_SERVER': bool(mail_server),
                    'MAIL_PORT': bool(mail_port),
                    'MAIL_USERNAME': bool(mail_username),
                    'MAIL_PASSWORD': bool(mail_password)
                }
            })
        
        # Try to connect to the mail server
        import smtplib
        server = None
        if app.config.get('MAIL_USE_TLS'):
            server = smtplib.SMTP(mail_server, mail_port)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(mail_server, mail_port)
        
        server.login(mail_username, mail_password)
        server.quit()
        
        return jsonify({
            'success': True,
            'message': 'Email configuration is valid'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error verifying email configuration: {str(e)}'
        })



# =============================================================================
# AUTHENTICATION ROUTES
# =============================================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    error = None
    if request.method == 'POST':
         username = request.form.get('username')
         password = request.form.get('password')
         user = User.query.filter_by(username=username).first()
         if user and user.check_password(password):
              session['user_id'] = user.id
              session['user_role'] = user.role  # Store role in session
              return redirect(url_for('index'))
         else:
              error = "Invalid username or password"
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    """Log out user by removing session data"""
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    """Allow users to reset their password"""
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Look up the user by username.
        user = User.query.filter_by(username=username).first()
        if not user:
            error = "Invalid username."
        elif not user.check_password(current_password):
            error = "Current password is incorrect."
        elif new_password != confirm_password:
            error = "New password and confirmation do not match."
        else:
            user.set_password(new_password)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('reset_password.html', error=error)


# =============================================================================
# WORK ORDER ROUTES
# =============================================================================
@app.route('/')
@login_required
def index():
    """Main page displaying open work orders"""
    sort_by = request.args.get('sort_by', 'id')
    order = request.args.get('order', 'asc')
    work_orders = get_sorted_work_orders("Open", sort_by, order)
    return render_template('index.html', work_orders=work_orders, sort_by=sort_by, order=order)


@app.route('/workorders/completed')
@login_required
def completed_work_orders():
    """Page displaying completed work orders"""
    sort_by = request.args.get('sort_by', 'id')
    order = request.args.get('order', 'asc')
    work_orders = get_sorted_work_orders("Complete", sort_by, order)
    return render_template('completed_work_orders.html', work_orders=work_orders, sort_by=sort_by, order=order)


@app.route('/workorders/closed')
@login_required
def closed_work_orders():
    """Page displaying closed work orders"""
    sort_by = request.args.get('sort_by', 'id')
    order = request.args.get('order', 'asc')
    work_orders = get_sorted_work_orders("Closed", sort_by, order)
    return render_template('closed_work_orders.html', work_orders=work_orders, sort_by=sort_by, order=order)


@app.route('/workorder/new', methods=['GET', 'POST'])
@login_required
def new_work_order():
    """Create a new work order"""
    if request.method == 'POST':
        customer_work_order_number = request.form.get('customer_work_order_number')
        rmj_job_number = request.form.get('rmj_job_number')
        
        # Check for duplicate RMJ Job Number
        existing_work_order = WorkOrder.query.filter_by(rmj_job_number=rmj_job_number).first()
        if existing_work_order:
            error = "A work order with that RMJ Job Number already exists."
            return render_template('new_work_order.html', error=error)
        
        description = request.form.get('description')
        status = request.form.get('status')
        owner = request.form.get('owner')
        estimated_hours = float(request.form.get('estimated_hours') or 0)
        priority = request.form.get('priority')
        location = request.form.get('location')
        scheduled_date = parse_date(request.form.get('scheduled_date'))
        classification = request.form.get('classification', 'Billable')
        approved_for_work = False
        current_user = User.query.get(session.get('user_id'))
        if current_user and (current_user.role.lower() == 'admin' or current_user.username.lower() == 'admin'):
            approved_for_work = bool(request.form.get('approved_for_work'))
        
        new_order = WorkOrder(
            customer_work_order_number=customer_work_order_number,
            rmj_job_number=rmj_job_number,
            description=description,
            status=status,
            owner=owner,
            estimated_hours=estimated_hours,
            priority=priority,
            location=location,
            scheduled_date=scheduled_date,
            classification=classification,
            approved_for_work=approved_for_work
        )
        db.session.add(new_order)
        db.session.commit()
        
        # Log the creation
        log_change(session.get('user_id'), "Created WorkOrder", "WorkOrder", new_order.id,
                   f"Created work order with RMJ Job Number: {rmj_job_number}")
        db.session.commit()
        
        # Send new work order notification
        send_new_work_order_notification(new_order)
        
        return redirect(url_for('index'))
    return render_template('new_work_order.html')


@app.route('/workorder/<int:work_order_id>')
@login_required
def work_order_detail(work_order_id):
    """View details of a specific work order"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    # Find any associated project
    project = Project.query.filter_by(work_order_id=work_order.id).first()
    
    default_engineer = ""
    if session.get('user_id'):
        current_user = User.query.get(session.get('user_id'))
        if current_user:
            default_engineer = get_engineer_name(current_user.username)
    
    # Get all users for the dropdown, ordered by full_name
    users = User.query.order_by(User.full_name.nulls_last(), User.username).all()
    
    return render_template('work_order_detail.html', 
                         work_order=work_order, 
                         default_engineer=default_engineer, 
                         associated_project=project,
                         users=users)


@app.route('/workorder/<int:work_order_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_work_order(work_order_id):
    """Edit an existing work order"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    if request.method == 'POST':
        old_status = work_order.status  # Store old status for notification
        
        work_order.customer_work_order_number = request.form.get('customer_work_order_number')
        work_order.rmj_job_number = request.form.get('rmj_job_number')
        work_order.description = request.form.get('description')
        work_order.status = request.form.get('status')
        work_order.owner = request.form.get('owner')
        work_order.estimated_hours = float(request.form.get('estimated_hours', work_order.estimated_hours))
        work_order.priority = request.form.get('priority')
        work_order.location = request.form.get('location')
        work_order.scheduled_date = parse_date(request.form.get('scheduled_date'))
        work_order.classification = request.form.get('classification', "Billable")
        work_order.requested_by = request.form.get('requested_by')
        current_user = User.query.get(session.get('user_id'))
        if current_user and (current_user.role.lower() == 'admin' or current_user.username.lower() == 'admin'):
            work_order.approved_for_work = bool(request.form.get('approved_for_work'))
        
        db.session.commit()
        
        # Log the edit
        log_change(session.get('user_id'), "Edited WorkOrder", "WorkOrder", work_order.id,
                   f"Edited work order with RMJ Job Number: {work_order.rmj_job_number}")
        db.session.commit()
        
        # Send status change notification if status changed
        if old_status != work_order.status:
            send_status_change_notification(work_order, old_status, work_order.status)
        
        return redirect(url_for('work_order_detail', work_order_id=work_order.id))
    return render_template('edit_work_order.html', work_order=work_order)

@app.route('/workorder/<int:work_order_id>/delete', methods=['POST'])
def delete_work_order(work_order_id):  # ← Fixed function name
    # Your existing delete function code here
    pass  # Remove this and add your actual delete code


@login_required
def delete_work_order(work_order_id):
    """Delete a work order"""
    password = request.form.get('password')
    if password != app.config.get('DELETE_PASSWORD'):
        return "Incorrect password", 403
    
    work_order = WorkOrder.query.get_or_404(work_order_id)
    rmj_job_number = work_order.rmj_job_number
    
    db.session.delete(work_order)
    db.session.commit()
    
    # Log deletion
    log_change(session.get('user_id'), "Deleted WorkOrder", "WorkOrder", work_order_id,
              f"Deleted work order with RMJ Job Number: {rmj_job_number}")
    db.session.commit()
    
    return redirect(url_for('index'))


@app.route('/workorder/<int:work_order_id>/download_report_template')
@login_required
def download_report_template(work_order_id):
    """Download a report template for a work order"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    return send_from_directory(
        'static', 
        'report_template.docx', 
        as_attachment=True, 
        download_name=f"WorkOrder_{work_order_id}_ReportTemplate.docx"
    )


@app.route('/search')
@login_required
def search():
    """Search for work orders by various criteria"""
    query = request.args.get('query', '')
    status = request.args.get('status', 'open')  # Default to 'open' if not specified
    
    # Determine the source page for the back button
    source_page = 'index'
    if status == 'closed':
        source_page = 'closed'
    elif status == 'completed':
        source_page = 'completed'
    elif status == 'all':
        source_page = 'all'
    
    if query:
        # Search by RMJ Job Number, Customer Work Order Number, or keywords in the description.
        search_filter = (
            WorkOrder.rmj_job_number.ilike(f'%{query}%') |
            WorkOrder.customer_work_order_number.ilike(f'%{query}%') |
            WorkOrder.description.ilike(f'%{query}%') |
            WorkOrder.owner.ilike(f'%{query}%') |
            WorkOrder.location.ilike(f'%{query}%')
        )
        
        # Apply status filter unless searching all
        if status == 'all':
            results = WorkOrder.query.filter(search_filter).all()
        else:
            results = WorkOrder.query.filter(search_filter & (WorkOrder.status == status)).all()
    else:
        results = []
    
    return render_template('search.html', query=query, results=results, source_page=source_page)


@app.route('/document/<int:document_id>/toggle_approval', methods=['POST'])
@login_required
def toggle_document_approval(document_id):
    """Toggle the approval status of a document"""
    document = WorkOrderDocument.query.get_or_404(document_id)
    
    # Get the updated approval status from the request
    data = request.get_json()
    is_approved = data.get('is_approved', False)
    
    # Update the document's approval status
    document.is_approved = is_approved
    
    # Log the change
    action = "Approved Document" if is_approved else "Unapproved Document"
    log_change(
        session.get('user_id'),
        action,
        "Document",
        document_id,
        f"{'Approved' if is_approved else 'Unapproved'} document {document.original_filename} for Work Order #{document.work_order_id}"
    )
    
    # Commit changes to the database
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/workorder/<int:work_order_id>/download_document/<int:document_id>')
@login_required
def download_document(work_order_id, document_id):
    # Fetch the WorkOrderDocument record (404 if not found)
    document = WorkOrderDocument.query.get_or_404(document_id)
    # Serve it from your UPLOAD_FOLDER with the original filename
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        document.filename,
        as_attachment=True,
        download_name=document.original_filename
    )

@app.route('/workorder/<int:work_order_id>/document/<int:document_id>/delete', methods=['POST'])
@login_required
def delete_work_order_document(work_order_id, document_id):
    # Look up the document (404 if not found)
    document = WorkOrderDocument.query.get_or_404(document_id)
    # Ensure it belongs to this work order
    if document.work_order_id != work_order_id:
        return "Invalid document for this work order", 400

    # Remove the file from disk
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete the DB record
    db.session.delete(document)
    db.session.commit()

    flash(f"Deleted document “{document.original_filename}”", "success")
    return redirect(url_for('work_order_detail', work_order_id=work_order_id))

    # Add this route to your app.py file in the routes section

@app.route('/work_order/<int:work_order_id>/upload_photos', methods=['POST'])
@login_required
def upload_photos(work_order_id):
    """Upload multiple photos for a work order"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    
    try:
        uploaded_files = request.files.getlist('photos')
        
        if not uploaded_files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        # Create upload directory if it doesn't exist
        upload_dir = app.config.get('UPLOAD_FOLDER', 'static/uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        uploaded_count = 0
        
        for file in uploaded_files:
            if file.filename == '':
                continue
                
            # Check file type
            if not file.content_type.startswith('image/'):
                continue
                
            # Generate unique filename
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name, ext = os.path.splitext(filename)
            unique_filename = f"wo_{work_order_id}_{timestamp}_{uploaded_count}_{name}{ext}"
            
            # Save file
            file_path = os.path.join(upload_dir, unique_filename)
            file.save(file_path)
            
            # Here you would typically save photo metadata to database
            # For now, we'll just count successful uploads
            uploaded_count += 1
        
        # Log the upload
        log_change(
            session.get('user_id'),
            "Uploaded Photos",
            "WorkOrder",
            work_order_id,
            f"Uploaded {uploaded_count} photos to Work Order #{work_order_id}"
        )
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'{uploaded_count} photos uploaded successfully',
            'uploaded_count': uploaded_count
        })
        
    except Exception as e:
        print(f"Error uploading photos: {e}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/uploads/<filename>')
def uploaded_ticket_photo(filename):
    """Serve uploaded photo files"""
    upload_dir = app.config.get('UPLOAD_FOLDER', 'static/uploads')
    return send_from_directory(upload_dir, filename)
        
# =============================================================================
# SIGNATURE ROUTES
# =============================================================================
@app.route('/workorder/<int:work_order_id>/add_signature', methods=['POST'])
@login_required
@rate_limit(max_calls=5, time_window=60)
def add_signature(work_order_id):
    """Add a signature to a work order"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    
    try:
        data = request.get_json()
        signee_name = data.get('signee_name', '').strip()
        signature_data = data.get('signature_data', '')
        
        if not signee_name:
            return jsonify({'success': False, 'message': 'Signee name is required'}), 400
        
        if not signature_data:
            return jsonify({'success': False, 'message': 'Signature is required'}), 400
        
        if not signature_data.startswith('data:image/'):
            return jsonify({'success': False, 'message': 'Invalid signature format'}), 400
        
        # Get client info for audit trail
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
        user_agent = request.headers.get('User-Agent', '')
        
        # Create new signature record
        new_signature = WorkOrderSignature(
            work_order_id=work_order_id,
            signee_name=signee_name,
            signature_data=signature_data,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        db.session.add(new_signature)
        db.session.commit()
        
        # Log the signature creation
        log_change(
            session.get('user_id'),
            "Added Signature",
            "WorkOrderSignature",
            new_signature.id,
            f"Signature added by {signee_name} for Work Order #{work_order.id}"
        )
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Signature saved successfully',
            'signature_id': new_signature.id,
            'signed_at': new_signature.signed_at.strftime('%Y-%m-%d %H:%M:%S'),
            'signee_name': new_signature.signee_name
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error saving signature: {e}")
        return jsonify({'success': False, 'message': 'Error saving signature'}), 500

@app.route('/workorder/<int:work_order_id>/signature/<int:signature_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_signature(work_order_id, signature_id):
    """Delete a signature (admin only)"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    signature = WorkOrderSignature.query.get_or_404(signature_id)
    
    if signature.work_order_id != work_order_id:
        return "Signature not found for this work order", 404
    
    signee_name = signature.signee_name
    
    log_change(
        session.get('user_id'),
        "Deleted Signature",
        "WorkOrderSignature",
        signature_id,
        f"Deleted signature by {signee_name} for Work Order #{work_order.id}"
    )
    
    db.session.delete(signature)
    db.session.commit()
    
    flash(f'Signature by {signee_name} has been deleted', 'success')
    return redirect(url_for('work_order_detail', work_order_id=work_order_id))

# =============================================================================
# TIME ENTRY ROUTES
# =============================================================================
@app.route('/workorder/<int:work_order_id>/add_time_inline', methods=['POST'])
@login_required
def add_time_inline(work_order_id):
    """Add a time entry directly from the work order detail page"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    # Prevent adding time if the work order is closed or complete.
    if work_order.status in ["Closed", "Complete"]:
        return "Cannot add time entries to a closed or complete work order.", 403

    # Get the engineer from the dropdown (user_id) or fallback to text input
    selected_user_id = request.form.get('engineer_user')
    if selected_user_id:
        selected_user = User.query.get(int(selected_user_id))
        engineer = get_engineer_name(selected_user.username) if selected_user else ""
    else:
        engineer = request.form.get('engineer')
        
    # If the engineer field is still empty, map the logged-in user's username to a full name.
    if not engineer or engineer.strip() == "":
        current_user = User.query.get(session.get('user_id'))
        if current_user:
            engineer = get_engineer_name(current_user.username)
        else:
            engineer = ""
    
    work_date_str = request.form.get('work_date')
    time_in_str = request.form.get('time_in')
    time_out_str = request.form.get('time_out')
    description = request.form.get('description')

    try:
        work_date = parse_date(work_date_str)
        time_in = parse_time(time_in_str)
        time_out = parse_time(time_out_str)
        if not all([work_date, time_in, time_out]):
            return "Invalid date or time format", 400
    except Exception:
        return "Invalid date or time format", 400

    hours_worked = calculate_hours(work_date, time_in, time_out)

    # GET CURRENT ROLE AND RATE - ADD THIS:
    current_role, current_rate = get_user_role_and_rate(engineer)

    new_entry = TimeEntry(
        work_order_id=work_order_id,
        engineer=engineer,
        work_date=work_date,
        time_in=time_in,
        time_out=time_out,
        hours_worked=hours_worked,
        description=description,
        role_at_time_of_entry=current_role,    # ADD THIS
        rate_at_time_of_entry=current_rate     # ADD THIS
    )
    db.session.add(new_entry)
    db.session.commit()

    # Log the creation of the time entry.
    log_change(
        session.get('user_id'),
        "Created TimeEntry",
        "TimeEntry",
        new_entry.id,
        f"Added time entry for engineer {engineer} with {hours_worked} hours on work order {work_order.rmj_job_number}"
    )
    db.session.commit()
    
    # Check hours threshold after adding time
    if work_order.estimated_hours > 0:
        hours_logged = work_order.hours_logged
        percentage = (hours_logged / work_order.estimated_hours) * 100
        
        # Check thresholds setting
        setting = get_notification_setting('hours_threshold')
        if setting:
            warning_threshold = setting.options.get('warning_threshold', 80)
            
            # Check if we've just crossed a threshold
            if (percentage >= warning_threshold and (percentage - (hours_worked / work_order.estimated_hours * 100)) < warning_threshold) or \
               (percentage >= 100 and (percentage - (hours_worked / work_order.estimated_hours * 100)) < 100):
                send_hours_threshold_notification(
                    work_order, 
                    hours_logged, 
                    work_order.estimated_hours, 
                    percentage
                )

    return redirect(url_for('work_order_detail', work_order_id=work_order.id))


@app.route('/time_entry/<int:time_entry_id>/delete', methods=['POST'])
@login_required
def delete_time_entry(time_entry_id):
    """Delete a time entry"""
    entry = TimeEntry.query.get_or_404(time_entry_id)
    
    # Check if entry is locked due to JL/JT checkboxes
    if entry.entered_on_jl or entry.entered_on_jt:
        flash("Cannot delete time entry: This entry has been entered into the accounting system (JL/JT checked).", "danger")
        return redirect(url_for('work_order_detail', work_order_id=entry.work_order_id))
    
    work_order_id = entry.work_order_id
    
    # Log the deletion
    log_change(
        session.get('user_id'),
        "Deleted TimeEntry",
        "TimeEntry",
        time_entry_id,
        f"Deleted time entry for engineer {entry.engineer} with {entry.hours_worked} hours"
    )
    
    db.session.delete(entry)
    db.session.commit()
    
    flash("Time entry deleted successfully.", "success")
    return redirect(url_for('work_order_detail', work_order_id=work_order_id))


@app.route('/time_entry/<int:time_entry_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_time_entry(time_entry_id):
    """Edit an existing time entry"""
    entry = TimeEntry.query.get_or_404(time_entry_id)
    
    # Check if entry is locked due to JL/JT checkboxes
    if entry.entered_on_jl or entry.entered_on_jt:
        flash("Cannot edit time entry: This entry has been entered into the accounting system (JL/JT checked).", "danger")
        return redirect(url_for('work_order_detail', work_order_id=entry.work_order_id))
    
    if request.method == 'POST':
        engineer = request.form.get('engineer')
        work_date_str = request.form.get('work_date')
        time_in_str = request.form.get('time_in')
        time_out_str = request.form.get('time_out')
        description = request.form.get('description')
        
        try:
            work_date = parse_date(work_date_str)
            time_in = parse_time(time_in_str)
            time_out = parse_time(time_out_str)
            if not all([work_date, time_in, time_out]):
                flash("Invalid date or time format", "danger")
                return render_template('edit_time_entry.html', entry=entry)
        except Exception as e:
            flash(f"Invalid date or time format: {e}", "danger")
            return render_template('edit_time_entry.html', entry=entry)
            
        entry.engineer = engineer
        entry.work_date = work_date
        entry.time_in = time_in
        entry.time_out = time_out
        entry.description = description
        entry.hours_worked = calculate_hours(work_date, time_in, time_out)
        
        # Log the edit
        log_change(
            session.get('user_id'),
            "Edited TimeEntry",
            "TimeEntry",
            entry.id,
            f"Edited time entry for engineer {entry.engineer}"
        )
        
        db.session.commit()
        flash("Time entry updated successfully.", "success")
        return redirect(url_for('work_order_detail', work_order_id=entry.work_order_id))
    
    return render_template('edit_time_entry.html', entry=entry)


@app.route('/time_entry/<int:time_entry_id>/reassign', methods=['GET', 'POST'])
@login_required
def reassign_time_entry(time_entry_id):
    """Reassign a time entry to a different work order"""
    entry = TimeEntry.query.get_or_404(time_entry_id)
    
    # Check if entry is locked due to JL/JT checkboxes
    if entry.entered_on_jl or entry.entered_on_jt:
        flash("Cannot reassign time entry: This entry has been entered into the accounting system (JL/JT checked).", "danger")
        return redirect(url_for('work_order_detail', work_order_id=entry.work_order_id))
    
    if request.method == 'POST':
        target_work_order_id = request.form.get('target_work_order_id')
        if not target_work_order_id:
            flash("Please select a target work order", "danger")
            work_orders = WorkOrder.query.all()
            return render_template('reassign_time_entry.html', entry=entry, work_orders=work_orders)
        
        old_work_order_id = entry.work_order_id
        entry.work_order_id = int(target_work_order_id)
        
        # Log the reassignment
        log_change(
            session.get('user_id'),
            "Reassigned TimeEntry",
            "TimeEntry",
            entry.id,
            f"Reassigned time entry from work order {old_work_order_id} to {target_work_order_id}"
        )
        
        db.session.commit()
        flash("Time entry reassigned successfully.", "success")
        return redirect(url_for('work_order_detail', work_order_id=entry.work_order_id))
    else:
        work_orders = WorkOrder.query.all()
        return render_template('reassign_time_entry.html', entry=entry, work_orders=work_orders)


@app.route('/time_entry/<int:entry_id>/update_checkboxes', methods=['POST'])
@login_required
@rate_limit(max_calls=10, time_window=10)
def update_time_entry_checkboxes(entry_id):
    """Update the checkbox status for a time entry"""
    try:
        entry = TimeEntry.query.get_or_404(entry_id)
        data = request.get_json()  # Expecting JSON data
        
        # Check if data was received
        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400
        
        # Update the checkboxes if data is provided
        if 'entered_on_jl' in data:
            entry.entered_on_jl = bool(data['entered_on_jl'])
        if 'entered_on_jt' in data:
            entry.entered_on_jt = bool(data['entered_on_jt'])
        
        # Commit the changes
        db.session.commit()
        
        # Return success response
        return jsonify({'success': True})
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error in update_time_entry_checkboxes: {str(e)}")
        db.session.rollback()  # Rollback any pending changes
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/workorder/<int:work_order_id>/export_time_entries')
@login_required
def export_time_entries_for_work_order(work_order_id):
    """Export time entries for a work order to Excel"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    time_entries = TimeEntry.query.filter_by(work_order_id=work_order.id).order_by(TimeEntry.work_date).all()
    
    if not time_entries:
        return "No time entries found for this work order.", 404

    data = []
    for entry in time_entries:
        data.append({
            "ID": entry.id,
            "Engineer": entry.engineer,
            "Work Date": entry.work_date.strftime('%Y-%m-%d'),
            "Time In": entry.time_in.strftime('%H:%M'),
            "Time Out": entry.time_out.strftime('%H:%M'),
            "Hours Worked": entry.hours_worked,
            "Description": entry.description,
            "Logged At": entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='TimeEntries')
    output.seek(0)

    filename = f"WorkOrder_{work_order.id}_TimeEntries.xlsx"
    return send_file(output, download_name=filename, as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.route('/time_entry/<int:entry_id>/update', methods=['POST'])
@login_required
def update_time_entry_ajax(entry_id):
    """Update a time entry via AJAX request"""
    entry = TimeEntry.query.get_or_404(entry_id)
    
    # Check if entry is locked due to JL/JT checkboxes
    if entry.entered_on_jl or entry.entered_on_jt:
        return jsonify({
            'success': False,
            'error': 'Cannot update time entry: This entry has been entered into the accounting system (JL/JT checked).'
        }), 403
    
    data = request.get_json()
    
    try:
        entry.engineer = data.get('engineer')
        entry.work_date = parse_date(data.get('work_date'))
        entry.time_in = parse_time(data.get('time_in')) 
        entry.time_out = parse_time(data.get('time_out'))
        entry.description = data.get('description')
        
        # Recalculate hours worked
        entry.hours_worked = calculate_hours(entry.work_date, entry.time_in, entry.time_out)
        
        # Log the update
        log_change(
            session.get('user_id'),
            "Updated TimeEntry via AJAX",
            "TimeEntry",
            entry.id,
            f"Updated time entry for engineer {entry.engineer}"
        )
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'hours_worked': entry.hours_worked
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Add this route to app.py in the TIME ENTRY ROUTES section

@app.route('/workorder/<int:work_order_id>/add_time_adjustment', methods=['POST'])
@login_required
@admin_required
def add_time_adjustment(work_order_id):
    """Add a time adjustment entry (admin only)"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    
    # Get the form fields from the adjustment form
    work_date_str = request.form.get('adjustment_work_date')
    hours_adjustment = request.form.get('hours_adjustment')
    description = request.form.get('adjustment_description')

    try:
        work_date = parse_date(work_date_str)
        hours_float = float(hours_adjustment)
        
        if not work_date:
            flash("Invalid date format", "danger")
            return redirect(url_for('work_order_detail', work_order_id=work_order_id))
            
        if hours_float == 0:
            flash("Hours adjustment cannot be zero", "warning")
            return redirect(url_for('work_order_detail', work_order_id=work_order_id))
            
    except (ValueError, TypeError):
        flash("Invalid hours format", "danger")
        return redirect(url_for('work_order_detail', work_order_id=work_order_id))

    # For time adjustments, we'll use dummy time values since they're required fields
    # but not meaningful for adjustments
    if hours_float > 0:
        # Positive adjustment: time_in = 09:00, time_out = 09:00 + hours
        time_in = time_class(9, 0)  # Use time_class instead of time
        hours_part = int(hours_float)
        minutes_part = int((hours_float % 1) * 60)
        time_out_dt = datetime.combine(work_date, time_in) + timedelta(hours=hours_part, minutes=minutes_part)
        time_out = time_out_dt.time()
    else:
        # Negative adjustment: time_in = 09:00, time_out = 09:00 (will show 0 hours but we'll override)
        time_in = time_class(9, 0)  # Use time_class instead of time
        time_out = time_class(9, 0)  # Use time_class instead of time

    new_entry = TimeEntry(
        work_order_id=work_order_id,
        engineer="timeadj",  # Hard-coded engineer name for adjustments
        work_date=work_date,
        time_in=time_in,
        time_out=time_out,
        hours_worked=hours_float,  # This can be negative
        description=description or "Time adjustment"
    )
    db.session.add(new_entry)
    db.session.commit()

    # Log the creation of the time adjustment
    log_change(
        session.get('user_id'),
        "Created Time Adjustment",
        "TimeEntry",
        new_entry.id,
        f"Added time adjustment of {hours_float} hours on work order {work_order.rmj_job_number}"
    )
    db.session.commit()
    
    # Flash appropriate message
    if hours_float > 0:
        flash(f"Added time adjustment of +{hours_float} hours", "success")
    else:
        flash(f"Added time adjustment of {hours_float} hours", "info")

    return redirect(url_for('work_order_detail', work_order_id=work_order.id))



# =============================================================================
# TIMESHEET ROUTES
# =============================================================================
@app.route('/timesheet/new', methods=['GET', 'POST'])
@login_required
def new_timesheet():
    """Create a new timesheet with multiple time entries"""
    default_engineer = ""
    current_user_id = None
    if session.get('user_id'):
        current_user = User.query.get(session.get('user_id'))
        if current_user:
            default_engineer = get_engineer_name(current_user.username)
            current_user_id = current_user.id
    
    if request.method == 'POST':
        entries_created = 0
        for i in range(1, 6):
            work_order_id = request.form.get(f'work_order_id_{i}')
            if work_order_id:
                wo = WorkOrder.query.get(int(work_order_id))
                if wo and wo.status in ["Closed", "Complete"]:
                    continue

                # Get the engineer from the dropdown (user_id) or fallback to text input
                selected_user_id = request.form.get(f'engineer_user_{i}')
                if selected_user_id:
                    selected_user = User.query.get(int(selected_user_id))
                    engineer = get_engineer_name(selected_user.username) if selected_user else default_engineer
                else:
                    engineer = request.form.get(f'engineer_{i}') or default_engineer
                
                work_date_str = request.form.get(f'work_date_{i}')
                time_in_str = request.form.get(f'time_in_{i}')
                time_out_str = request.form.get(f'time_out_{i}')
                description = request.form.get(f'description_{i}')
                
                try:
                    work_date = parse_date(work_date_str)
                    time_in = parse_time(time_in_str)
                    time_out = parse_time(time_out_str)
                    if not all([work_date, time_in, time_out]):
                        continue
                except Exception:
                    continue
        
                current_role, current_rate = get_user_role_and_rate(engineer)

                hours_worked = calculate_hours(work_date, time_in, time_out)
                new_entry = TimeEntry(
                    work_order_id=int(work_order_id),
                    engineer=engineer,
                    work_date=work_date,
                    time_in=time_in,
                    time_out=time_out,
                    hours_worked=hours_worked,
                    description=description,
                    role_at_time_of_entry=current_role,   
                    rate_at_time_of_entry=current_rate     
                )
                db.session.add(new_entry)
                entries_created += 1
        db.session.commit()
        if entries_created > 0:
            return redirect(url_for('index'))
        else:
            return "No valid entries submitted", 400
    else:
        work_orders = WorkOrder.query.all()
        # Get all users for the dropdown, ordered by full_name
        users = User.query.order_by(User.full_name.nulls_last(), User.username).all()
        
        return render_template('timesheet_new.html', 
                             work_orders=work_orders, 
                             default_engineer=default_engineer,
                             current_user_id=current_user_id,
                             users=users)


@app.route('/timesheet/select', methods=['GET', 'POST'])
@login_required
def select_weekly_timesheet():
    """Select a weekly timesheet to view"""
    if request.method == 'POST':
        year = request.form.get('year')
        week = request.form.get('week')
        try:
            year = int(year)
            week = int(week)
        except ValueError:
            return "Invalid input", 400
        return redirect(url_for('view_weekly_timesheet', year=year, week=week))
    else:
        current_year = date.today().year
        try:
            year = int(request.args.get('year', current_year))
        except ValueError:
            year = current_year

        jan1 = date(year, 1, 1)
        offset = (jan1.weekday() + 1) % 7
        first_sunday = jan1 - timedelta(days=offset)

        week_options = []
        for n in range(1, 53):
            week_sunday = first_sunday + timedelta(days=(n - 1) * 7)
            week_saturday = week_sunday + timedelta(days=6)
            option_text = f"Week {n}: {week_sunday.strftime('%Y-%m-%d')} to {week_saturday.strftime('%Y-%m-%d')}"
            week_options.append((n, option_text))
        
        return render_template('select_timesheet.html', year=year, week_options=week_options)


@app.route('/timesheet/weeks/<int:year>')
@login_required
def get_week_options(year):
    """Get week options for a specific year"""
    jan1 = date(year, 1, 1)
    offset = (jan1.weekday() + 1) % 7
    first_sunday = jan1 - timedelta(days=offset)
    
    week_options = []
    for n in range(1, 53):
        week_sunday = first_sunday + timedelta(days=(n - 1) * 7)
        week_saturday = week_sunday + timedelta(days=6)
        option_text = f"Week {n}: {week_sunday.strftime('%Y-%m-%d')} to {week_saturday.strftime('%Y-%m-%d')}"
        week_options.append({'week': n, 'text': option_text})
    return jsonify(week_options)


@app.route('/timesheet/<int:year>/<int:week>')
@login_required
def view_weekly_timesheet(year, week):
    """View a weekly timesheet"""
    start_date, last_date = get_week_dates(year, week)
    if not start_date or not last_date:
        return "Invalid year/week combination", 400
    
    sort_by = request.args.get('sort_by', 'work_date')
    order = request.args.get('order', 'asc')
    
    valid_sort_columns = {
        'work_date': TimeEntry.work_date,
        'engineer': TimeEntry.engineer,
    }
    sort_column = valid_sort_columns.get(sort_by, TimeEntry.work_date)
    sort_order = sort_column.asc() if order == 'asc' else sort_column.desc()
    
    engineer_filter = request.args.get('engineer', None)
    query = TimeEntry.query.filter(TimeEntry.work_date >= start_date, TimeEntry.work_date <= last_date)
    if engineer_filter:
        query = query.filter(TimeEntry.engineer.ilike(f'%{engineer_filter}%'))
    time_entries = query.order_by(sort_order).all()
    total_hours = sum(entry.hours_worked for entry in time_entries)
    
    return render_template('weekly_timesheet.html',
                           year=year,
                           week=week,
                           start_date=start_date,
                           last_date=last_date,
                           time_entries=time_entries,
                           total_hours=total_hours,
                           sort_by=sort_by,
                           order=order)


@app.route('/timesheet/export/<int:year>/<int:week>')
@login_required
def export_timesheet(year, week):
    """Export a weekly timesheet to Excel"""
    start_date, last_date = get_week_dates(year, week)
    if not start_date or not last_date:
        return "Invalid year/week combination", 400
    
    query = TimeEntry.query.filter(
        TimeEntry.work_date >= start_date,
        TimeEntry.work_date <= last_date
    )
    
    engineer_filter = request.args.get('engineer', None)
    if engineer_filter:
        query = query.filter(TimeEntry.engineer.ilike(f'%{engineer_filter}%'))
    
    time_entries = query.order_by(TimeEntry.work_date).all()

    if not time_entries:
        return "No time entries found for this query.", 404

    data = []
    for entry in time_entries:
        data.append({
            'ID': entry.id,
            'Engineer': entry.engineer,
            'Work Order': f"{entry.work_order.rmj_job_number} - {entry.work_order.customer_work_order_number}",
            'Date': entry.work_date.strftime('%Y-%m-%d'),
            'Time In': entry.time_in.strftime('%H:%M'),
            'Time Out': entry.time_out.strftime('%H:%M'),
            'Hours Worked': entry.hours_worked,
            'Description': entry.description,
        })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Timesheet')
    output.seek(0)
    
    filename = f"Timesheet_{year}_week{week}.xlsx"
    return send_file(
        output,
        download_name=filename,
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/timesheet/ajax/entries_by_date')
@login_required
def get_entries_by_date():
    """API endpoint to get time entries for a specific date"""
    try:
        # Get the date parameter
        date_str = request.args.get('date')
        year = int(request.args.get('year'))
        week = int(request.args.get('week'))
        
        # Parse the date
        entry_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Get all time entries for this date within the current filter context
        query = TimeEntry.query.filter(TimeEntry.work_date == entry_date)
        
        # Apply any filters that were applied to the main view
        engineer_filter = request.args.get('engineer', None)
        if engineer_filter:
            query = query.filter(TimeEntry.engineer.ilike(f'%{engineer_filter}%'))
            
        # Get the entries
        entries = query.all()
        
        # Format the entries for JSON response
        result = []
        for entry in entries:
            result.append({
                'id': entry.id,
                'engineer': entry.engineer,
                'time_in': entry.time_in.strftime('%H:%M'),
                'time_out': entry.time_out.strftime('%H:%M'),
                'hours_worked': float(entry.hours_worked),
                'work_order': entry.work_order.rmj_job_number,
                'description': entry.description,
                'classification': entry.work_order.classification
            })
        
        return jsonify({'success': True, 'entries': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================================================
# DOCUMENT UPLOAD ROUTES
# =============================================================================
@app.route('/workorder/<int:work_order_id>/upload_document', methods=['GET', 'POST'])
@login_required
def upload_document(work_order_id):
    work_order = WorkOrder.query.get_or_404(work_order_id)
    # Get associated project if it exists
    associated_project = Project.query.filter_by(work_order_id=work_order_id).first()
    
    if request.method == 'POST':
        if 'document' not in request.files:
            return "No file part", 400
        file = request.files['document']
        if file.filename == '':
            return "No file selected", 400
        filename = secure_filename(file.filename)
        
        # Create upload folder if it doesn't exist
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        # Save file locally
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        # Get document type from form
        document_type = request.form.get('document_type', 'regular')
        
        # Create document record with local file path
        document = WorkOrderDocument(
            work_order_id=work_order_id,
            project_id=associated_project.id if associated_project else None,
            filename=filename,
            original_filename=file.filename,
            upload_time=datetime.utcnow(),
            document_type=document_type
        )
        
        db.session.add(document)
        db.session.commit()
        
        # Check if notifications are enabled and determine document type
        is_report = (document_type == 'report') or is_report_file(file.filename)
        
        # Send appropriate notifications
        notification_sent = False
        
        if is_report:
            # Send report upload notification
            if send_report_notification(work_order, document):
                notification_sent = True
                flash("Report uploaded and notification sent.", "success")
            
            # Send report approval notification if enabled
            approval_setting = get_notification_setting('report_approval')
            if approval_setting:
                if send_report_approval_notification(work_order, document):
                    flash("Report approval notification sent.", "success")
                else:
                    flash("Report approval notification failed to send.", "warning")
        
        if not notification_sent:
            flash("Document uploaded successfully.", "success")
            
        return redirect(url_for('work_order_detail', work_order_id=work_order_id))
        
    return render_template('upload_document.html', work_order=work_order)


# =============================================================================
# IMPORT/EXPORT ROUTES
# =============================================================================
@app.route('/import', methods=['GET', 'POST'])
@login_required
def import_excel():
    """Import work orders from Excel"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        try:
            df = pd.read_excel(file)
            for index, row in df.iterrows():
                scheduled_date = None
                if 'scheduled_date' in row and not pd.isna(row['scheduled_date']):
                    try:
                        scheduled_date = pd.to_datetime(row['scheduled_date']).date()
                    except Exception:
                        scheduled_date = None
                work_order = WorkOrder(
                    customer_work_order_number=row.get('customer_work_order_number', ''),
                    rmj_job_number=row.get('rmj_job_number', ''),
                    description=row.get('description', ''),
                    status=row.get('status', ''),
                    owner=row.get('owner', ''),
                    estimated_hours=float(row.get('estimated_hours', 0)),
                    priority=row.get('priority', 'Medium'),
                    location=row.get('location', ''),
                    scheduled_date=scheduled_date
                )
                db.session.add(work_order)
            db.session.commit()
            return redirect(url_for('index'))
        except Exception as e:
            return f"Error processing file: {e}", 400
    return render_template('import.html')


@app.route('/export')
@login_required
def export_excel():
    """Export all work orders to Excel"""
    work_orders = WorkOrder.query.all()
    data = []
    for wo in work_orders:
        data.append({
            "ID": wo.id,
            "Customer Work Order Number": wo.customer_work_order_number,
            "RMJ Job Number": wo.rmj_job_number,
            "Description": wo.description,
            "Status": wo.status,
            "Owner": wo.owner,
            "Estimated Hours": wo.estimated_hours,
            "Hours Logged": wo.hours_logged,
            "Hours Remaining": wo.hours_remaining,
            "Priority": wo.priority,
            "Location": wo.location,
            "Scheduled Date": wo.scheduled_date.strftime('%Y-%m-%d') if wo.scheduled_date else ''
        })
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='WorkOrders')
    output.seek(0)
    return send_file(output, download_name="workorders.xlsx", as_attachment=True, 
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


# =============================================================================
# PROJECT ROUTES
# =============================================================================
@app.route('/projects')
@login_required
def projects():
    """View all projects"""
    projects = Project.query.all()
    # Filter for only projects related to Contract/Project work orders
    contract_projects = []
    for project in projects:
        if project.work_order and project.work_order.classification == 'Contract/Project':
            contract_projects.append(project)
    return render_template('project_dashboard.html', projects=contract_projects)


@app.route('/projects/new', methods=['GET', 'POST'])
@login_required
def new_project():
    """Create a new project"""
    if request.method == 'POST':
        work_order_id = request.form.get('work_order_id')
        name = request.form.get('name')
        description = request.form.get('description')
        start_date = parse_date(request.form.get('start_date'))
        end_date = parse_date(request.form.get('end_date'))
        
        # Validate the work order exists and is a Contract/Project type
        work_order = WorkOrder.query.get_or_404(int(work_order_id))
        if work_order.classification != 'Contract/Project':
            return "Only Contract/Project work orders can have associated projects", 400
        
        # Check if this work order already has a project
        existing_project = Project.query.filter_by(work_order_id=work_order_id).first()
        if existing_project:
            return "This work order already has an associated project", 400
        
        new_project = Project(
            work_order_id=work_order_id,
            name=name,
            description=description,
            start_date=start_date,
            end_date=end_date,
            status='Planning'
        )
        db.session.add(new_project)
        db.session.commit()
        
        # Log the creation
        log_change(session.get('user_id'), "Created Project", "Project", new_project.id,
                   f"Created project '{name}' for Work Order #{work_order_id}")
        db.session.commit()
        
        return redirect(url_for('project_detail', project_id=new_project.id))
    
    # Get all Contract/Project work orders that don't already have projects
    work_orders = WorkOrder.query.filter_by(classification='Contract/Project').all()
    eligible_work_orders = []
    for wo in work_orders:
        if not Project.query.filter_by(work_order_id=wo.id).first():
            eligible_work_orders.append(wo)
            
    return render_template('new_project.html', work_orders=eligible_work_orders)


@app.route('/projects/<int:project_id>')
@login_required
def project_detail(project_id):
    """View project details"""
    project = Project.query.get_or_404(project_id)
    
    # Get the default engineer name for the form
    default_engineer = ""
    if session.get('user_id'):
        current_user = User.query.get(session.get('user_id'))
        if current_user:
            default_engineer = get_engineer_name(current_user.username)
    
    # Today's date for the form
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get all users for the assigned_to dropdown
    users = User.query.order_by(User.full_name).all()
    
    return render_template(
        'project_detail.html', 
        project=project, 
        default_engineer=default_engineer, 
        today_date=today_date,
        users=users
    )


@app.route('/projects/<int:project_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_project(project_id):
    """Edit a project"""
    project = Project.query.get_or_404(project_id)
    if request.method == 'POST':
        project.name = request.form.get('name')
        project.description = request.form.get('description')
        project.start_date = parse_date(request.form.get('start_date'))
        project.end_date = parse_date(request.form.get('end_date'))
        project.status = request.form.get('status')
        
        db.session.commit()
        
        # Log the edit
        log_change(session.get('user_id'), "Edited Project", "Project", project.id,
                   f"Edited project '{project.name}'")
        db.session.commit()
        
        return redirect(url_for('project_detail', project_id=project.id))
    return render_template('edit_project.html', project=project)


@app.route('/projects/<int:project_id>/delete', methods=['POST'])
@login_required
def delete_project(project_id):
    """Delete a project"""
    project = Project.query.get_or_404(project_id)
    project_name = project.name
    
    # Unlink any time entries from tasks in this project
    for task in project.tasks:
        for entry in task.time_entries:
            entry.task_id = None
    
    # Delete the project (will cascade delete tasks due to relationship)
    db.session.delete(project)
    
    # Log the deletion
    log_change(
        session.get('user_id'),
        "Deleted Project",
        "Project",
        project_id,
        f"Deleted project '{project_name}' and all associated tasks"
    )
    
    db.session.commit()
    
    return redirect(url_for('projects'))


@app.route('/projects/<int:project_id>/upload_document', methods=['GET', 'POST'])
@login_required
def upload_project_document(project_id):
    """Upload a document for a project"""
    project = Project.query.get_or_404(project_id)
    work_order = WorkOrder.query.get_or_404(project.work_order_id)
    
    if request.method == 'POST':
        if 'document' not in request.files:
            return "No file part", 400
        file = request.files['document']
        if file.filename == '':
            return "No file selected", 400
        filename = secure_filename(file.filename)
        
        # Create upload folder if it doesn't exist
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        # Get document type from form
        document_type = request.form.get('document_type', 'regular')
        
        # Create document linked to both work order and project
        document = WorkOrderDocument(
            work_order_id=project.work_order_id,
            project_id=project_id,
            filename=filename,
            original_filename=file.filename,
            upload_time=datetime.utcnow(),
            document_type=document_type
        )
        db.session.add(document)
        db.session.commit()
        
        return redirect(url_for('project_detail', project_id=project_id))
        
    # Change this line to use the existing template and pass both project and work_order
    return render_template('upload_document.html', work_order=work_order, project=project)


@app.route('/projects/<int:project_id>/document/<int:document_id>/delete', methods=['POST'])
@login_required
def delete_project_document(project_id, document_id):
    """Delete a project document"""
    document = WorkOrderDocument.query.get_or_404(document_id)
    
    # Verify the document belongs to this project
    if document.project_id != project_id:
        return "Invalid document", 400
        
    # Store information for logging
    doc_name = document.original_filename
    
    # Delete the document
    db.session.delete(document)
    
    # Log the deletion
    log_change(
        session.get('user_id'), 
        "Deleted Project Document", 
        "WorkOrderDocument",
        document_id,
        f"Deleted document '{doc_name}' from project"
    )
    
    db.session.commit()
    
    return redirect(url_for('project_detail', project_id=project_id))


# =============================================================================
# PROJECT TASK ROUTES
# =============================================================================
@app.route('/projects/<int:project_id>/task/new', methods=['GET', 'POST'])
@login_required
def new_task(project_id):
    """Create a new task in a project"""
    project = Project.query.get_or_404(project_id)
    
    # Get all users for the dropdown
    users = User.query.order_by(User.full_name).all()
    
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        start_date = parse_date(request.form.get('start_date'))
        end_date = parse_date(request.form.get('end_date'))
        estimated_hours = float(request.form.get('estimated_hours') or 0)
        priority = request.form.get('priority')
        dependencies = request.form.get('dependencies')
        assigned_to = request.form.get('assigned_to')
        
        new_task = ProjectTask(
            project_id=project_id,
            name=name,
            description=description,
            start_date=start_date,
            end_date=end_date,
            estimated_hours=estimated_hours,
            priority=priority,
            dependencies=dependencies,
            assigned_to=assigned_to
        )
        db.session.add(new_task)
        db.session.commit()
        
        # Log the creation
        log_change(session.get('user_id'), "Created Task", "ProjectTask", new_task.id,
                   f"Created task '{name}' for Project #{project_id}")
        db.session.commit()
        
        return redirect(url_for('project_detail', project_id=project_id))
        
    return render_template('new_task.html', project=project, existing_tasks=project.tasks, users=users)


@app.route('/projects/<int:project_id>/quick_add_task', methods=['POST'])
@login_required
def quick_add_task(project_id):
    """Quickly add a task to a project"""
    project = Project.query.get_or_404(project_id)
    
    try:
        # Get form data
        name = request.form.get('name')
        start_date = parse_date(request.form.get('start_date'))
        end_date = parse_date(request.form.get('end_date'))
        estimated_hours = float(request.form.get('estimated_hours') or 0)
        priority = request.form.get('priority', 'Medium')
        assigned_to = request.form.get('assigned_to', '')
        
        # Create new task
        new_task = ProjectTask(
            project_id=project_id,
            name=name,
            description="",  # Empty description for quick add
            start_date=start_date,
            end_date=end_date,
            estimated_hours=estimated_hours,
            status="Not Started",  # Default status
            priority=priority,
            assigned_to=assigned_to
        )
        
        # Get the highest position value for tasks in this project
        highest_position = db.session.query(db.func.max(ProjectTask.position)).filter_by(project_id=project_id).scalar() or 0
        new_task.position = highest_position + 1
        
        db.session.add(new_task)
        
        # Log the creation
        log_change(
            session.get('user_id'),
            "Created Task via Quick Add",
            "ProjectTask",
            None,  # Will be updated after commit
            f"Quick added task '{name}' for Project #{project_id}"
        )
        
        db.session.commit()
        
        # Update the log entry with the new task's ID
        log_entry = ChangeLog.query.filter_by(
            action="Created Task via Quick Add",
            object_type="ProjectTask",
            description=f"Quick added task '{name}' for Project #{project_id}"
        ).order_by(ChangeLog.timestamp.desc()).first()
        
        if log_entry:
            log_entry.object_id = new_task.id
            db.session.commit()
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': True,
                'task_id': new_task.id,
                'name': new_task.name
            })
        
        # If not AJAX, redirect to the project detail page
        return redirect(url_for('project_detail', project_id=project_id))
        
    except Exception as e:
        db.session.rollback()
        
        # If AJAX request, return JSON error
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
            
        # If not AJAX, redirect with error
        flash(f"Error creating task: {str(e)}", "error")
        return redirect(url_for('project_detail', project_id=project_id))


@app.route('/projects/tasks/<int:task_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_task(task_id):
    """Edit a project task"""
    task = ProjectTask.query.get_or_404(task_id)
    project = task.project
    
    # Get all users for the dropdown
    users = User.query.order_by(User.full_name).all()
    
    if request.method == 'POST':
        task.name = request.form.get('name')
        task.description = request.form.get('description')
        task.start_date = parse_date(request.form.get('start_date'))
        task.end_date = parse_date(request.form.get('end_date'))
        task.estimated_hours = float(request.form.get('estimated_hours') or 0)
        task.status = request.form.get('status')
        task.priority = request.form.get('priority')
        task.dependencies = request.form.get('dependencies')
        task.assigned_to = request.form.get('assigned_to')
        
        # Get and set the progress percentage
        progress_percent = request.form.get('progress_percent')
        if progress_percent is not None:
            try:
                task.progress_percent = int(progress_percent)
            except ValueError:
                # If conversion fails, don't update the field
                pass
                
        # If status is completed, ensure progress is 100%
        if task.status == 'Completed':
            task.progress_percent = 100
        
        db.session.commit()
        
        # Log the edit
        log_change(session.get('user_id'), "Edited Task", "ProjectTask", task.id,
                   f"Edited task '{task.name}' for Project #{project.id}")
        db.session.commit()
        
        return redirect(url_for('project_detail', project_id=project.id))
        
    return render_template('edit_task.html', task=task, project=project, existing_tasks=project.tasks, users=users)


@app.route('/projects/tasks/<int:task_id>/update_hours', methods=['POST'])
@login_required
def update_task_hours(task_id):
    """Update task hours and create time entry"""
    task = ProjectTask.query.get_or_404(task_id)
    project = task.project
    actual_hours = float(request.form.get('actual_hours') or 0)
    
    # Check if this is the first time hours are being added
    is_first_time_entry = task.actual_hours == 0 and actual_hours > 0
    
    # Update task hours
    task.actual_hours += actual_hours
    
    # Update task status based on completion if needed
    if task.actual_hours >= task.estimated_hours and task.status != 'Completed':
        task.status = 'Completed'
    # If this is the first time hours are being logged and status is 'Not Started', set to 'In Progress'
    elif is_first_time_entry and task.status == 'Not Started':
        task.status = 'In Progress'
        log_change(session.get('user_id'), "Updated Task Status", "ProjectTask", task.id,
                  f"Automatically updated task '{task.name}' status to 'In Progress' after logging first hours")
    
    # Add a time entry for this work
# Add a time entry for this work
    if actual_hours > 0:
        # Create a time entry linked to both the work order and the task
        engineer = request.form.get('engineer')
        work_date = parse_date(request.form.get('work_date'))
        description = request.form.get('description') or f"Work on task: {task.name}"
        
        # Calculate time_in and time_out based on hours
        time_in_str = request.form.get('time_in')
        time_out_str = request.form.get('time_out')
        time_in = parse_time(time_in_str)
        time_out = parse_time(time_out_str)
        actual_hours = calculate_hours(work_date, time_in, time_out)  # Recalculate from times
        
        # GET CURRENT ROLE AND RATE - ADD THIS:
        current_role, current_rate = get_user_role_and_rate(engineer)

        time_entry = TimeEntry(
            work_order_id=project.work_order_id,
            task_id=task.id,  # Link to task
            engineer=engineer,
            work_date=work_date,
            time_in=time_in,
            time_out=time_out,
            hours_worked=actual_hours,
            description=description,
            role_at_time_of_entry=current_role,    # ADD THIS
            rate_at_time_of_entry=current_rate     # ADD THIS
        )
        db.session.add(time_entry)
    
    db.session.commit()
    
    # Log the hours update
    log_change(session.get('user_id'), "Updated Task Hours", "ProjectTask", task.id,
               f"Added {actual_hours} hours to task '{task.name}' for Project #{project.id}")
    db.session.commit()
    
    return redirect(url_for('project_detail', project_id=project.id))


@app.route('/projects/tasks/<int:task_id>/reset_hours', methods=['POST'])
@login_required
def reset_task_hours(task_id):
    """Reset hours for a task"""
    task = ProjectTask.query.get_or_404(task_id)
    project = task.project
    
    # Get the submitted actual hours value
    new_actual_hours = float(request.form.get('actual_hours') or 0)
    
    # Update the task's actual hours
    task.actual_hours = new_actual_hours
    
    # Log the hours update
    log_change(
        session.get('user_id'),
        "Reset Task Hours",
        "ProjectTask",
        task.id,
        f"Reset hours for task '{task.name}' from {task.actual_hours} to {new_actual_hours}"
    )
    
    db.session.commit()
    
    return redirect(url_for('project_detail', project_id=project.id))


@app.route('/projects/tasks/<int:task_id>/delete', methods=['POST'])
@login_required
def delete_task(task_id):
    """Delete a task"""
    task = ProjectTask.query.get_or_404(task_id)
    project_id = task.project_id
    task_name = task.name
    
    # Check if there are any time entries associated with this task
    time_entries = TimeEntry.query.filter_by(task_id=task_id).all()
    
    # For each time entry, remove the task_id reference (don't delete the time entries)
    for entry in time_entries:
        entry.task_id = None
    
    # Delete the task
    db.session.delete(task)
    
    # Log the deletion
    log_change(
        session.get('user_id'),
        "Deleted Task",
        "ProjectTask",
        task_id,
        f"Deleted task '{task_name}' from Project #{project_id}"
    )
    
    db.session.commit()
    
    return redirect(url_for('project_detail', project_id=project_id))


@app.route('/projects/tasks/<int:task_id>/time_entries')
@login_required
def task_time_entries(task_id):
    """View time entries for a task"""
    task = ProjectTask.query.get_or_404(task_id)
    project = task.project
    work_order = project.work_order
    
    # Get time entries directly linked to this task
    task_entries = TimeEntry.query.filter_by(task_id=task_id).order_by(TimeEntry.work_date.desc()).all()
    
    # Get unassigned time entries from the work order
    unassigned_entries = TimeEntry.query.filter_by(
        work_order_id=work_order.id, 
        task_id=None
    ).order_by(TimeEntry.work_date.desc()).all()
    
    # Get default engineer and today's date
    default_engineer = ""
    if session.get('user_id'):
        current_user = User.query.get(session.get('user_id'))
        if current_user:
            default_engineer = get_engineer_name(current_user.username)
    
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    return render_template(
        'task_time_entries.html',
        task=task,
        project=project,
        work_order=work_order,
        task_entries=task_entries,
        unassigned_entries=unassigned_entries,
        default_engineer=default_engineer,
        today_date=today_date
    )


@app.route('/time_entry/<int:entry_id>/assign_to_task/<int:task_id>', methods=['POST'])
@login_required
def assign_time_entry_to_task(entry_id, task_id):
    """Assign a time entry to a task"""
    entry = TimeEntry.query.get_or_404(entry_id)
    task = ProjectTask.query.get_or_404(task_id)
    project = task.project
    
    # Make sure the time entry belongs to the same work order as the task's project
    if entry.work_order_id != project.work_order_id:
        error_msg = "Time entry does not belong to this project's work order"
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': error_msg}), 400
        flash(error_msg, "danger")
        return redirect(url_for('task_time_entries', task_id=task_id))
    
    # Assign the time entry to the task
    entry.task_id = task.id
    
    # Update the task's actual hours
    task.actual_hours += entry.hours_worked
    
    # Update task status if needed
    if task.actual_hours >= task.estimated_hours and task.status != 'Completed':
        task.status = 'In Progress'
    
    db.session.commit()
    
    # Log the assignment
    log_change(
        session.get('user_id'),
        "Assigned Time Entry",
        "TimeEntry",
        entry.id,
        f"Assigned time entry #{entry.id} to task '{task.name}' in project #{project.id}"
    )
    db.session.commit()
    
    # Check if this is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': True,
            'entry_id': entry.id,
            'task_id': task.id,
            'actual_hours': task.actual_hours,
            'hours_remaining': task.hours_remaining
        })
    
    # If not AJAX, use the standard redirect
    next_page = request.args.get('next') or url_for('task_time_entries', task_id=task.id)
    flash("Time entry assigned to task successfully.", "success")
    return redirect(next_page)




@app.route('/time_entry/<int:entry_id>/remove_from_task/<int:task_id>', methods=['POST'])
@login_required
def remove_time_entry_from_task(entry_id, task_id):
    """Remove a time entry from a task"""
    entry = TimeEntry.query.get_or_404(entry_id)
    task = ProjectTask.query.get_or_404(task_id)
    
    # Make sure the time entry is actually assigned to this task
    if entry.task_id != task.id:
        error_msg = "Time entry not assigned to this task"
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': error_msg}), 400
        return error_msg, 400
    
    # Subtract hours from the task's actual hours
    task.actual_hours -= entry.hours_worked
    if task.actual_hours < 0:
        task.actual_hours = 0
    
    # Unassign the time entry from the task
    entry.task_id = None
    
    db.session.commit()
    
    # Log the removal
    log_change(
        session.get('user_id'),
        "Removed Time Entry",
        "TimeEntry",
        entry.id,
        f"Removed time entry #{entry.id} from task '{task.name}'"
    )
    db.session.commit()
    
    # Check if this is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': True,
            'entry_id': entry.id,
            'task_id': task.id,
            'actual_hours': task.actual_hours,
            'hours_remaining': task.hours_remaining
        })
    
    # If not AJAX, use the standard redirect
    return redirect(url_for('task_time_entries', task_id=task.id))


# =============================================================================
# GANTT CHART ROUTES
# =============================================================================
@app.route('/projects/<int:project_id>/gantt')
@login_required
def project_gantt(project_id):
    """View the Gantt chart for a project"""
    project = Project.query.get_or_404(project_id)
    return render_template('project_gantt.html', project=project)


@app.route('/api/projects/<int:project_id>/gantt_data')
@login_required
def project_gantt_data(project_id):
    """API endpoint to provide data for the Gantt chart"""
    project = Project.query.get_or_404(project_id)
    
    # Get tasks ordered by position field
    tasks = ProjectTask.query.filter_by(project_id=project_id).order_by(ProjectTask.position).all()
    
    # Debug information
    print(f"Project: {project.name}, Tasks count: {len(tasks)}")
    
    # Prepare data in DHTMLX Gantt format
    data = []
    links = []
    link_id = 1
    
    for task in tasks:
        # Skip tasks without dates
        if not task.start_date or not task.end_date:
            continue
            
        # Calculate duration in days
        duration = (task.end_date - task.start_date).days
        if duration <= 0:
            duration = 1  # Minimum duration is 1 day
        
        # Format task for Gantt chart - round progress to 2 decimal places
        progress_value = round(task.completion_percentage / 100, 2)
        
        task_data = {
            "id": task.id,
            "text": task.name,
            "start_date": task.start_date.strftime('%Y-%m-%d'),
            "duration": duration,
            "progress": progress_value,  # Rounded to 2 decimal places
            "status": task.status,
            "open": True,
            "sort_order": task.position  # Include position for reference
        }
        
        # Add custom data if needed
        if task.description:
            task_data["description"] = task.description
        if task.assigned_to:
            task_data["assigned_to"] = task.assigned_to
            
        data.append(task_data)
        
        # Add dependency links if any
        if task.dependencies:
            try:
                dependency_ids = [int(dep.strip()) for dep in task.dependencies.split(',') if dep.strip()]
                
                for dep_id in dependency_ids:
                    links.append({
                        "id": link_id,
                        "source": dep_id,
                        "target": task.id,
                        "type": "0"  # Finish-to-Start dependency type
                    })
                    link_id += 1
            except Exception as e:
                print(f"Error processing dependencies for task {task.id}: {e}")
    
    # Log the data we're returning
    print(f"Returning {len(data)} tasks and {len(links)} links")
    
    # Return the data in the format expected by DHTMLX Gantt
    return jsonify({
        "data": data,
        "links": links
    })


@app.route('/api/projects/<int:project_id>/reorder_tasks', methods=['POST'])
@login_required
def reorder_project_tasks(project_id):
    """API endpoint to save the new order of tasks after drag and drop"""
    project = Project.query.get_or_404(project_id)
    
    # Get data from request
    data = request.json
    task_id = data.get('task_id')
    new_index = data.get('new_index')
    tasks_order = data.get('tasks_order')
    
    if not all([task_id, isinstance(new_index, int), tasks_order]):
        return jsonify({'success': False, 'error': 'Invalid data provided'}), 400
    
    try:
        # Get the task that was moved
        task = ProjectTask.query.get_or_404(task_id)
        
        # Update the positions of all tasks in the project according to their new order
        for i, task_id in enumerate(tasks_order):
            current_task = ProjectTask.query.get(task_id)
            if current_task and current_task.project_id == project_id:
                current_task.position = i
        
        # Log the reordering
        log_change(
            session.get('user_id'),
            "Reordered Tasks",
            "Project",
            project_id,
            f"Reordered task '{task.name}' to position {new_index} in project '{project.name}'"
        )
        
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    

@app.route('/api/projects/<int:project_id>/create_task', methods=['POST'])
@login_required
def create_task_from_gantt(project_id):
    """API endpoint to create a new task from the Gantt chart view"""
    project = Project.query.get_or_404(project_id)
    
    try:
        # Get task data from request
        data = request.json
        task_name = data.get('text', 'New Task')  # Task name/text
        start_date_str = data.get('start_date')
        duration = int(data.get('duration', 1))
        
        # Convert start_date to datetime
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else date.today()
        
        # Calculate end_date based on duration
        end_date = start_date + timedelta(days=duration)
        
        # Create new task
        new_task = ProjectTask(
            project_id=project_id,
            name=task_name,
            description="Created from Gantt chart",
            start_date=start_date,
            end_date=end_date,
            estimated_hours=8.0,  # Default estimated hours
            priority="Medium",  # Default priority
            status="Not Started"  # Default status
        )
        
        # Get the highest position value for tasks in this project
        highest_position = db.session.query(db.func.max(ProjectTask.position)).filter_by(project_id=project_id).scalar() or 0
        new_task.position = highest_position + 1
        
        db.session.add(new_task)
        
        # Log the creation
        log_change(
            session.get('user_id'),
            "Created Task from Gantt",
            "ProjectTask",
            None,  # Will be updated after commit
            f"Created task '{task_name}' from Gantt chart for Project #{project_id}"
        )
        
        db.session.commit()
        
        # Update the log entry with the new task's ID
        log_entry = ChangeLog.query.filter_by(
            action="Created Task from Gantt",
            object_type="ProjectTask",
            description=f"Created task '{task_name}' from Gantt chart for Project #{project_id}"
        ).order_by(ChangeLog.timestamp.desc()).first()
        
        if log_entry:
            log_entry.object_id = new_task.id
            db.session.commit()
        
        # Return the created task with its ID
        return jsonify({
            'success': True,
            'id': new_task.id,
            'text': new_task.name,
            'start_date': new_task.start_date.strftime('%Y-%m-%d'),
            'duration': duration,
            'progress': 0
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/tasks/<int:task_id>/update', methods=['POST'])
@login_required
def update_task_from_gantt(task_id):
    """API endpoint to update a task from the Gantt chart view"""
    task = ProjectTask.query.get_or_404(task_id)
    
    try:
        # Get data from request
        data = request.json
        
        # Update task properties
        if 'text' in data:
            task.name = data['text']
        
        if 'start_date' in data:
            task.start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
        
        if 'duration' in data and task.start_date:
            # Calculate end_date based on duration
            task.end_date = task.start_date + timedelta(days=int(data['duration']))
        
        if 'progress' in data:
            # Convert progress (0-1) to percentage (0-100)
            progress_percentage = int(float(data['progress']) * 100)
            task.progress_percent = progress_percentage
            
            # Update status based on progress if appropriate
            if progress_percentage == 100 and task.status != 'Completed':
                task.status = 'Completed'
            elif progress_percentage > 0 and task.status == 'Not Started':
                task.status = 'In Progress'
                
        # Log the update
        log_change(
            session.get('user_id'),
            "Updated Task from Gantt",
            "ProjectTask",
            task.id,
            f"Updated task '{task.name}' from Gantt chart"
        )
        
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# ADMIN ROUTES
# =============================================================================


@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Admin users page"""
    try:
        if not session.get('user_id'):
            return redirect(url_for('login'))
        
        users = User.query.all()
        return render_template('admin_users.html', users=users, worker_roles=[])
    except Exception as e:
        print(f"Error: {e}")
        return redirect(url_for('index'))

@app.route('/admin/users/<int:user_id>/assign_worker_role', methods=['POST'])
@login_required
@admin_required
def admin_assign_worker_role(user_id):
    """Assign worker role to user"""
    if not session.get('user_id'):
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    
    current_user = User.query.get(session.get('user_id'))
    if not current_user or (current_user.role.lower() != 'admin' and current_user.username.lower() != 'admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    try:
        user = User.query.get_or_404(user_id)
        worker_role_id = request.form.get('worker_role_id')
        
        if worker_role_id:
            # Assign worker role
            worker_role = WorkerRole.query.get(worker_role_id)
            if worker_role:
                user.assigned_role_id = worker_role_id
                flash(f'Assigned {worker_role.name} position to {user.username}', 'success')
            else:
                flash('Invalid worker role selected', 'error')
        else:
            # Remove worker role assignment
            user.assigned_role_id = None
            flash(f'Removed worker position from {user.username}', 'success')
        
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating worker role: {str(e)}', 'error')
    
    return redirect(url_for('admin_users'))



@app.route('/admin/users/new', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_new_user():
    """Create a new user"""
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password')
        role = request.form.get('role', 'user').strip().lower()
        full_name = request.form.get('full_name', '').strip()
        
        # Force the role to "admin" if the username is "admin"
        if username.lower() == "admin":
            role = "admin"
        
        # Check for duplicate username
        if User.query.filter_by(username=username).first():
            return "User already exists", 400
        
        new_user = User(username=username, role=role, full_name=full_name)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('admin_users'))
    return render_template('admin_new_user.html')


@app.route('/admin/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_user(user_id):
    """Edit a user"""
    user = User.query.get_or_404(user_id)
    if request.method == 'POST':
        username = request.form.get('username').strip()
        full_name = request.form.get('full_name', '').strip()
        
        # If editing the admin account, force role to "admin"
        if username.lower() == "admin":
            role = "admin"
        else:
            role = request.form.get('role', 'user').strip().lower()
        new_password = request.form.get('password')
        
        user.username = username
        user.role = role
        user.full_name = full_name
        if new_password:
            user.set_password(new_password)
        db.session.commit()
        return redirect(url_for('admin_users'))
    return render_template('admin_edit_user.html', user=user)


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    """Delete a user"""
    user = User.query.get_or_404(user_id)
    # Prevent deletion of the admin account itself.
    if user.username.lower() == "admin":
        return "Cannot delete admin user", 403
    db.session.delete(user)
    db.session.commit()
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/update_role', methods=['POST'])
@login_required
@admin_required
def admin_update_user_role(user_id):
    """Update a user's role"""
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role', 'user').lower()  # Ensure lowercase
    
    # Force admin account to remain admin
    if user.username.lower() == 'admin':
        new_role = 'admin'
    
    # Set and commit role change
    user.role = new_role
    db.session.commit()
    
    # If updating own role, refresh session
    if user_id == session.get('user_id'):
        # You may need to refresh the session
        session['user_role'] = new_role  # Add role to session
        
    return redirect(url_for('admin_users'))

@app.route('/admin_edit_rate')
@login_required
@admin_required
def admin_edit_rate():
    """Admin page to manage worker roles and rates"""
    try:
        # Check if user is logged in
        if not session.get('user_id'):
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        
        # Get current user
        current_user = User.query.get(session.get('user_id'))
        if not current_user:
            flash('User not found.', 'error')
            return redirect(url_for('login'))
        
        # Check if admin
        is_admin = (current_user.role and current_user.role.lower() == 'admin') or (current_user.username and current_user.username.lower() == 'admin')
        
        if not is_admin:
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('index'))
        
        # Get worker roles from database
        roles = WorkerRole.query.filter_by(is_active=True).order_by(WorkerRole.hourly_rate.desc()).all()
        
        # Check if worker_role_id field exists
        has_worker_role_field = False
        sample_user = User.query.first()
        if sample_user:
            has_worker_role_field = hasattr(sample_user, 'worker_role_id')
        
        # Create a dictionary of role assignments for the overview
        role_assignments = {}
        unassigned_users = []
        
        if has_worker_role_field:
            for role in roles:
                role_assignments[role.id] = {
                    'role': role,
                    'users': [],
                    'user_count': 0
                }
                
                # Get users assigned to this role
                try:
                    users_in_role = User.query.filter_by(worker_role_id=role.id).order_by(User.full_name.nulls_last(), User.username).all()
                    role_assignments[role.id]['users'] = users_in_role
                    role_assignments[role.id]['user_count'] = len(users_in_role)
                except Exception as e:
                    print(f"Error querying users for role {role.id}: {e}")
                    role_assignments[role.id]['users'] = []
            
            # Get unassigned users
            try:
                unassigned_users = User.query.filter(
                    (User.worker_role_id.is_(None)) | (User.worker_role_id == 0)
                ).order_by(User.full_name.nulls_last(), User.username).all()
            except Exception as e:
                print(f"Error querying unassigned users: {e}")
        
        return render_template('admin_edit_rate.html', 
                             roles=roles,
                             role_assignments=role_assignments,
                             unassigned_users=unassigned_users,
                             has_worker_role_field=has_worker_role_field)
        
    except Exception as e:
        print(f"Exception in admin_edit_rate route: {e}")
        import traceback
        traceback.print_exc()
        flash('An error occurred loading the page.', 'error')
        return redirect(url_for('index'))

@app.route('/admin_edit_rate/add_role', methods=['POST'])
@login_required
@admin_required
def add_worker_role_from_edit_page():
    """Add new worker role from the admin_edit_rate page"""
    print("=== DEBUG: add_worker_role_from_edit_page called ===")
    
    try:
        # Admin check
        if not session.get('user_id'):
            print("DEBUG: No user_id in session")
            return jsonify({'error': 'Unauthorized'}), 401
        
        current_user = User.query.get(session.get('user_id'))
        if not current_user:
            print("DEBUG: Current user not found")
            return jsonify({'error': 'User not found'}), 401
        
        is_admin = (current_user.role and current_user.role.lower() == 'admin') or (current_user.username and current_user.username.lower() == 'admin')
        if not is_admin:
            print("DEBUG: User is not admin")
            return jsonify({'error': 'Access denied'}), 403
        
        # Get form data
        name = request.form.get('name', '').strip()
        hourly_rate = request.form.get('hourly_rate', '0')
        
        print(f"DEBUG: Received name='{name}', hourly_rate='{hourly_rate}'")
        
        if not name:
            print("DEBUG: Name is empty")
            return jsonify({'error': 'Role name is required'}), 400
        
        try:
            hourly_rate = float(hourly_rate)
            print(f"DEBUG: Converted hourly_rate to float: {hourly_rate}")
        except ValueError:
            print("DEBUG: Invalid hourly rate format")
            return jsonify({'error': 'Invalid hourly rate'}), 400
        
        if hourly_rate <= 0:
            print("DEBUG: Hourly rate is <= 0")
            return jsonify({'error': 'Hourly rate must be greater than 0'}), 400
        
        # Create the role directly using the WorkerRole class defined in this file
        try:
            print("DEBUG: Checking if role already exists")
            # Check if role already exists
            existing_role = WorkerRole.query.filter_by(name=name, is_active=True).first()
            if existing_role:
                print(f"DEBUG: Role already exists with ID {existing_role.id}")
                return jsonify({'error': 'A role with this name already exists'}), 400
            
            print("DEBUG: Creating new role")
            # Create new role directly
            new_role = WorkerRole(name=name, hourly_rate=hourly_rate, is_active=True)
            db.session.add(new_role)
            
            print("DEBUG: Committing to database")
            db.session.commit()
            
            print(f"DEBUG: Role created successfully with ID {new_role.id}")
            
            response_data = {
                'success': True,
                'role': {
                    'id': new_role.id,
                    'name': new_role.name,
                    'rate': float(new_role.hourly_rate)
                }
            }
            print(f"DEBUG: Returning response: {response_data}")
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"DEBUG: Database error: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return jsonify({'error': f'Database error occurred: {str(e)}'}), 500
        
    except Exception as e:
        print(f"DEBUG: General error in add_worker_role_from_edit_page: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error occurred: {str(e)}'}), 500

@app.route('/admin_edit_rate/remove_role/<int:role_id>', methods=['POST'])
@login_required
@admin_required
def remove_worker_role_from_edit_page(role_id):
    """Remove worker role from the admin_edit_rate page"""
    try:
        # Admin check
        if not session.get('user_id'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        current_user = User.query.get(session.get('user_id'))
        if not current_user:
            return jsonify({'error': 'User not found'}), 401
        
        is_admin = (current_user.role and current_user.role.lower() == 'admin') or (current_user.username and current_user.username.lower() == 'admin')
        if not is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        # Try to remove the role
        try:
            # Check if WorkerRole class is available
            if 'WorkerRole' not in globals():
                return jsonify({'error': 'WorkerRole class not available'}), 500
            
            # Find the role
            role = WorkerRole.query.get(role_id)
            if not role:
                return jsonify({'error': 'Role not found'}), 404
            
            # Mark as inactive instead of deleting
            role.is_active = False
            db.session.commit()
            
            return jsonify({'success': True})
            
        except Exception as e:
            db.session.rollback()
            print(f"Database error: {e}")
            return jsonify({'error': f'Database error occurred: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Error removing role: {e}")
        return jsonify({'error': f'Server error occurred: {str(e)}'}), 500
    
@app.route('/admin/changelog')
@login_required
@admin_required
def admin_changelog():
    """View the change log"""
    logs = ChangeLog.query.order_by(ChangeLog.timestamp.desc()).all()
    return render_template('admin_changelog.html', logs=logs)

@app.route('/admin/roles')
@login_required
@admin_required
def admin_roles():
    """Admin page to view and manage worker roles"""
    # Check if user is admin
    if not session.get('user_id'):
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session.get('user_id'))
    if not user or (user.role.lower() != 'admin' and user.username.lower() != 'admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    # Get all active roles
    roles = WorkerRole.query.filter_by(is_active=True).all()
    return render_template('admin_edit_rate.html', roles=roles)

@app.route('/admin/roles/add', methods=['POST'])
@login_required
@admin_required
def add_worker_role():
    """Add a new worker role"""
    # Admin check code here...

    name = request.form.get('name', '').strip()
    hourly_rate = float(request.form.get('hourly_rate', 0))

    try:
        from models import WorkerRole
        new_role = WorkerRole(name=name, hourly_rate=hourly_rate)
        db.session.add(new_role)
        db.session.commit()

        # Get updated roles list (assuming you need this)
        roles = WorkerRole.query.all()

        return jsonify({
            'success': True,
            'roles': [
                {
                    'id': role.id,
                    'name': role.name,
                    'rate': float(role.hourly_rate)
                }
                for role in roles
            ]
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error adding role: {e}")
        flash(f"Failed to add role '{name}': {e}", 'danger')
        return redirect(url_for('admin_roles'))

@app.route('/admin/roles/<int:role_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_worker_role(role_id):
    """Delete a worker role"""
    # Admin check and deletion code here...
    pass

@app.route('/admin_edit_rate/edit_role/<int:role_id>', methods=['POST'])
@login_required
@admin_required
def edit_worker_role_rate(role_id):
    """Edit the hourly rate for a worker role"""
    try:
        # Admin check
        if not session.get('user_id'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        current_user = User.query.get(session.get('user_id'))
        if not current_user:
            return jsonify({'error': 'User not found'}), 401
        
        is_admin = (current_user.role and current_user.role.lower() == 'admin') or (current_user.username and current_user.username.lower() == 'admin')
        if not is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        # Get the role
        role = WorkerRole.query.get(role_id)
        if not role:
            return jsonify({'error': 'Role not found'}), 404
        
        # Get form data
        new_name = request.form.get('name', '').strip()
        new_rate = request.form.get('hourly_rate', '0')
        
        if not new_name:
            return jsonify({'error': 'Role name is required'}), 400
        
        try:
            new_rate = float(new_rate)
        except ValueError:
            return jsonify({'error': 'Invalid hourly rate'}), 400
        
        if new_rate <= 0:
            return jsonify({'error': 'Hourly rate must be greater than 0'}), 400
        
        # Check if name already exists (excluding current role)
        existing_role = WorkerRole.query.filter(
            WorkerRole.name == new_name,
            WorkerRole.is_active == True,
            WorkerRole.id != role_id
        ).first()
        if existing_role:
            return jsonify({'error': 'A role with this name already exists'}), 400
        
        # Update the role
        old_name = role.name
        old_rate = role.hourly_rate
        role.name = new_name
        role.hourly_rate = new_rate
        db.session.commit()
        
        # Log the change
        log_change(
            session.get('user_id'),
            "Updated Worker Role",
            "WorkerRole",
            role.id,
            f"Updated role '{old_name}' (${old_rate}/hr) to '{new_name}' (${new_rate}/hr)"
        )
        db.session.commit()
        
        return jsonify({
            'success': True,
            'role': {
                'id': role.id,
                'name': role.name,
                'rate': float(role.hourly_rate)
            }
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error editing role: {e}")
        return jsonify({'error': f'Server error occurred: {str(e)}'}), 500
    
@app.route('/admin_edit_rate/assign_users/<int:role_id>')
@login_required
@admin_required
def assign_users_to_role(role_id):
    """Show page to assign users to a specific worker role"""
    try:
        # Get the role
        role = WorkerRole.query.get_or_404(role_id)
        
        # Get all users ordered by full_name, then username
        all_users = User.query.order_by(User.full_name.nulls_last(), User.username).all()
        
        # Check if worker_role_id field exists by testing with first user
        has_worker_role_field = False
        if all_users:
            first_user = all_users[0]
            has_worker_role_field = hasattr(first_user, 'worker_role_id')
        
        # Get users currently assigned to this role and unassigned users
        assigned_users = []
        unassigned_users = []
        
        for user in all_users:
            # Do all the hasattr checking here instead of in template
            user_has_role_field = has_worker_role_field and hasattr(user, 'worker_role_id')
            user_assigned_to_role = user_has_role_field and user.worker_role_id == role_id
            
            # Add additional properties to user object for template use
            user.has_worker_role_field = user_has_role_field
            user.is_assigned_to_current_role = user_assigned_to_role
            
            if user_assigned_to_role:
                assigned_users.append(user)
            else:
                unassigned_users.append(user)
        
        # Get all roles for the overview section
        all_roles = WorkerRole.query.filter_by(is_active=True).order_by(WorkerRole.name).all()
        
        # Create a dictionary of role assignments for the overview
        role_assignments = {}
        for role_item in all_roles:
            role_assignments[role_item.id] = {
                'role': role_item,
                'users': []
            }
            
            # Get users assigned to this role (only if field exists)
            if has_worker_role_field:
                try:
                    users_in_role = []
                    for user in User.query.filter_by(worker_role_id=role_item.id).order_by(User.full_name.nulls_last(), User.username).all():
                        # Add the same properties for consistency
                        user.has_worker_role_field = True
                        user.is_assigned_to_current_role = (user.worker_role_id == role_id)
                        users_in_role.append(user)
                    role_assignments[role_item.id]['users'] = users_in_role
                except Exception as e:
                    print(f"Error querying users for role {role_item.id}: {e}")
                    role_assignments[role_item.id]['users'] = []
        
        return render_template('assign_users_to_role.html',
                             role=role,
                             assigned_users=assigned_users,
                             unassigned_users=unassigned_users,
                             all_roles=all_roles,
                             role_assignments=role_assignments,
                             has_worker_role_field=has_worker_role_field)
        
    except Exception as e:
        print(f"Error in assign_users_to_role: {e}")
        import traceback
        traceback.print_exc()
        flash(f'An error occurred loading the assignment page: {str(e)}', 'error')
        return redirect(url_for('admin_edit_rate'))

@app.route('/admin_edit_rate/assign_user', methods=['POST'])
@login_required
@admin_required
def assign_user_to_worker_role():
    """Assign or unassign a user to/from a worker role"""
    try:
        user_id = request.form.get('user_id')
        role_id = request.form.get('role_id')  # Can be None to unassign
        action = request.form.get('action')  # 'assign' or 'unassign'
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Get the user
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user has worker_role_id field
        if not hasattr(user, 'worker_role_id'):
            return jsonify({'error': 'Database not ready. Please ensure the worker_role_id column has been added to the User table. You may need to restart the application after running the database migration.'}), 500
        
        old_role_id = user.worker_role_id
        old_role_name = "No Role"
        if old_role_id:
            old_role = WorkerRole.query.get(old_role_id)
            if old_role:
                old_role_name = old_role.name
        
        if action == 'assign' and role_id:
            # Get the role to assign
            role = WorkerRole.query.get(role_id)
            if not role:
                return jsonify({'error': 'Role not found'}), 404
            
            user.worker_role_id = int(role_id)
            new_role_name = role.name
            
            # Log the assignment
            log_change(
                session.get('user_id'),
                "Assigned Worker Role",
                "User",
                user.id,
                f"Assigned {user.full_name or user.username} from '{old_role_name}' to '{new_role_name}'"
            )
            
        elif action == 'unassign':
            user.worker_role_id = None
            new_role_name = "No Role"
            
            # Log the unassignment
            log_change(
                session.get('user_id'),
                "Unassigned Worker Role",
                "User",
                user.id,
                f"Unassigned {user.full_name or user.username} from '{old_role_name}'"
            )
        else:
            return jsonify({'error': 'Invalid action or missing role ID'}), 400
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Successfully updated {user.full_name or user.username} to: {new_role_name}',
            'user_id': user.id,
            'new_role_id': user.worker_role_id,
            'user_display_name': user.full_name or user.username
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error in assign_user_to_worker_role: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error occurred: {str(e)}'}), 500



@app.route('/admin_edit_rate/bulk_assign', methods=['POST'])
@login_required
@admin_required
def bulk_assign_users():
    """Bulk assign multiple users to a role"""
    try:
        user_ids = request.form.getlist('user_ids')
        role_id = request.form.get('role_id')
        action = request.form.get('action')  # 'assign' or 'unassign'
        
        if not user_ids:
            return jsonify({'error': 'No users selected'}), 400
        
        if action == 'assign' and not role_id:
            return jsonify({'error': 'Role is required for assignment'}), 400
        
        # Check if user model supports worker roles
        sample_user = User.query.first()
        if not sample_user or not hasattr(sample_user, 'worker_role_id'):
            return jsonify({'error': 'Database not ready. Please ensure the worker_role_id column has been added to the User table.'}), 500
        
        updated_users = []
        role_name = "No Role"
        
        if action == 'assign' and role_id:
            role = WorkerRole.query.get(role_id)
            if not role:
                return jsonify({'error': 'Role not found'}), 404
            role_name = role.name
        
        # Update each user
        for user_id in user_ids:
            user = User.query.get(user_id)
            if user:
                old_role_id = user.worker_role_id
                old_role_name = "No Role"
                if old_role_id:
                    old_role = WorkerRole.query.get(old_role_id)
                    if old_role:
                        old_role_name = old_role.name
                
                if action == 'assign':
                    user.worker_role_id = int(role_id)
                else:  # unassign
                    user.worker_role_id = None
                
                updated_users.append(user.full_name or user.username)
                
                # Log the change
                log_change(
                    session.get('user_id'),
                    f"Bulk {action.title()} Worker Role",
                    "User",
                    user.id,
                    f"{action.title()}ed {user.full_name or user.username} from '{old_role_name}' to '{role_name}'"
                )
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Successfully {action}ed {len(updated_users)} users to: {role_name}',
            'updated_count': len(updated_users)
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error in bulk_assign_users: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error occurred: {str(e)}'}), 500
    
@app.route('/admin/debug/check_user_model')
@login_required
@admin_required
def debug_check_user_model():
    """Debug route to check if User model has worker_role_id field"""
    try:
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        
        # Check table structure
        columns = inspector.get_columns('user')
        column_names = [col['name'] for col in columns]
        
        # Check if field exists in model
        sample_user = User.query.first()
        has_field_in_model = hasattr(sample_user, 'worker_role_id') if sample_user else False
        
        # Get sample data
        user_count = User.query.count()
        role_count = WorkerRole.query.count()
        
        result = f"""
        <h2>User Model Debug Information</h2>
        <p><strong>User table columns:</strong> {column_names}</p>
        <p><strong>worker_role_id in database:</strong> {'worker_role_id' in column_names}</p>
        <p><strong>worker_role_id in model:</strong> {has_field_in_model}</p>
        <p><strong>Total users:</strong> {user_count}</p>
        <p><strong>Total worker roles:</strong> {role_count}</p>
        
        <h3>Sample Users:</h3>
        <ul>
        """
        
        users = User.query.limit(5).all()
        for user in users:
            worker_role_info = "No worker_role_id field"
            if has_field_in_model:
                if user.worker_role_id:
                    role = WorkerRole.query.get(user.worker_role_id)
                    worker_role_info = f"Role: {role.name if role else 'Unknown'}"
                else:
                    worker_role_info = "No role assigned"
            
            result += f"<li>{user.full_name or user.username} - {worker_role_info}</li>"
        
        result += "</ul>"
        result += f'<p><a href="{url_for("admin_edit_rate")}">Back to Roles Management</a></p>'
        
        return result
        
    except Exception as e:
        import traceback
        return f"<h2>Error:</h2><pre>{traceback.format_exc()}</pre>"

@app.route('/admin/email_settings', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_email_settings():
    """Manage email notification settings"""
    if request.method == 'POST':
        # Process global notification settings
        global_enabled = 'notification_enabled' in request.form
        default_email = request.form.get('default_notification_email', '')
        
        # Debug output - before processing
        print("==== FORM DATA RECEIVED ====")
        for key, value in request.form.items():
            print(f"{key}: {value}")
        
        # Update all notification types
        notification_types = [
            'report_upload', 'report_approval', 'status_change', 
            'hours_threshold', 'scheduled_date', 'new_work_order'
        ]
        
        # Update each notification type
        for notification_type in notification_types:
            setting = NotificationSetting.query.filter_by(notification_type=notification_type).first()
            if not setting:
                # Create if it doesn't exist
                setting = NotificationSetting(notification_type=notification_type)
                db.session.add(setting)
            
            # Only enable if global notifications are enabled and this type is checked
            type_enabled = f'{notification_type}_enabled' in request.form
            setting.enabled = global_enabled and type_enabled
            
            # Get or create options dictionary
            options = {}
            
            # Process type-specific options
            if notification_type == 'report_upload':
                keywords = request.form.get('report_keywords', '')
                options['report_keywords'] = [k.strip() for k in keywords.split(',') if k.strip()]
            
            elif notification_type == 'report_approval':
                options['send_reminder'] = 'report_approval_reminder' in request.form
                options['reminder_days'] = int(request.form.get('report_approval_reminder_days', 3))
            
            elif notification_type == 'status_change':
                options['open_to_complete'] = 'status_open_to_complete' in request.form
                options['complete_to_closed'] = 'status_complete_to_closed' in request.form
                options['any_to_open'] = 'status_any_to_open' in request.form
            
            elif notification_type == 'hours_threshold':
                options['warning_threshold'] = int(request.form.get('hours_warning_threshold', 80))
                options['exceeded_alert'] = 'hours_exceeded_alert' in request.form
                options['include_work_order_owner'] = 'include_work_order_owner' in request.form
            
            elif notification_type == 'scheduled_date':
                options['days_before'] = int(request.form.get('scheduled_date_days', 3))
                options['include_owner'] = 'scheduled_include_owner' in request.form
            
            elif notification_type == 'new_work_order':
                options['high_priority'] = 'new_work_order_high' in request.form
                options['medium_priority'] = 'new_work_order_medium' in request.form
                options['low_priority'] = 'new_work_order_low' in request.form
            
            setting.options = options
            
            # Update recipients for this notification type
            # FIXED: Use the format that's actually coming from the form
            recipient_key = f"{notification_type}_recipients"
            recipient_emails = request.form.getlist(recipient_key)
            
            print(f"Recipients for {notification_type}: {recipient_emails}")
            
            # Clear existing recipients
            NotificationRecipient.query.filter_by(notification_setting_id=setting.id).delete()
            
            # Add new recipients
            for email in recipient_emails:
                if email and '@' in email:
                    recipient = NotificationRecipient(
                        notification_setting_id=setting.id,
                        email=email.strip()
                    )
                    db.session.add(recipient)
        
        # Store the default email in app config for redundancy
        app.config['REPORT_NOTIFICATION_EMAIL'] = default_email
        app.config['REPORT_NOTIFICATION_ENABLED'] = global_enabled
        
        db.session.commit()
        flash('Email settings updated successfully', 'success')
        
        # CHANGE: Don't redirect, continue to load the template with fresh data
    
    # Prepare data for the template - will be used for both GET and after POST
    notification_settings = {}
    report_keywords_list = []
    
    # Load settings from database
    for setting in NotificationSetting.query.all():
        notification_settings[setting.notification_type] = {
            'enabled': setting.enabled,
            'options': setting.options,
            'recipients': [r.email for r in setting.recipients]
        }
        
        # Extract report keywords for the template
        if setting.notification_type == 'report_upload' and setting.options and 'report_keywords' in setting.options:
            report_keywords_list = setting.options['report_keywords']
    
    # Get global enabled status from report_upload setting
    global_enabled = False
    report_upload = NotificationSetting.query.filter_by(notification_type='report_upload').first()
    if report_upload:
        global_enabled = report_upload.enabled
    
    # Prepare template data with values from database
    template_data = {
        'notification_enabled': global_enabled,
        'notification_email': app.config.get('REPORT_NOTIFICATION_EMAIL', ''),
        'mail_server': app.config.get('MAIL_SERVER', ''),
        'mail_port': app.config.get('MAIL_PORT', 587),
        'mail_use_tls': app.config.get('MAIL_USE_TLS', True),
        'mail_username': app.config.get('MAIL_USERNAME', ''),
        'report_keywords': ','.join(report_keywords_list),
        'settings': notification_settings
    }
    
    # Debug output - what's being sent to template
    print("==== TEMPLATE DATA ====")
    print(f"notification_enabled: {template_data['notification_enabled']}")
    print(f"notification_email: {template_data['notification_email']}")
    print(f"report_keywords: {template_data['report_keywords']}")
    print("Settings:")
    for key, value in notification_settings.items():
        print(f"  {key}: enabled={value['enabled']}, recipients={value['recipients']}")
    
    return render_template('admin_email_settings.html', **template_data)

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard with analytics"""
    try:
        # Get time frame from request parameters, default to last 30 days
        time_frame = request.args.get('time_frame', 'month')
        
        # Calculate start and end dates based on time frame
        end_date = datetime.now().date()
        if time_frame == 'week':
            start_date = end_date - timedelta(days=7)
        elif time_frame == 'month':
            start_date = end_date - timedelta(days=30)
        elif time_frame == 'quarter':
            start_date = end_date - timedelta(days=90)
        elif time_frame == 'year':
            start_date = end_date - timedelta(days=365)
        else:
            # Custom date range
            start_date_str = request.args.get('start_date')
            end_date_str = request.args.get('end_date')
            start_date = parse_date(start_date_str) or (end_date - timedelta(days=30))
            end_date = parse_date(end_date_str) or end_date
        
        # Get all time entries in the selected time period
        time_entries = TimeEntry.query.filter(
            TimeEntry.work_date >= start_date,
            TimeEntry.work_date <= end_date
        ).all()
        
        # Get all work orders associated with these time entries
        work_order_ids = set(entry.work_order_id for entry in time_entries if entry.work_order_id is not None)
        work_orders = WorkOrder.query.filter(WorkOrder.id.in_(work_order_ids)).all() if work_order_ids else []
        
        # Create a mapping of work order ID to classification
        work_order_classification = {}
        for wo in work_orders:
            # If classification field exists
            if hasattr(wo, 'classification') and wo.classification is not None:
                classification = wo.classification
            else:
                # Temporary classification logic based on existing fields
                if wo.customer_work_order_number and wo.customer_work_order_number.strip():
                    classification = 'Contract/Project'
                elif wo.description and ('internal' in wo.description.lower() or 'non-billable' in wo.description.lower()):
                    classification = 'Non-Billable'
                else:
                    classification = 'Billable'
            work_order_classification[wo.id] = classification
        
        # Get all engineers who have time entries in the selected period
        engineers = sorted(list(set(entry.engineer for entry in time_entries if entry.engineer is not None)))
        
        # Initialize data structures
        engineer_hours = {engineer: 0 for engineer in engineers}
        classification_hours = {'Contract/Project': 0, 'Billable': 0, 'Non-Billable': 0}
        engineer_classification_hours = {
            engineer: {'Contract/Project': 0, 'Billable': 0, 'Non-Billable': 0} 
            for engineer in engineers
        }
        
        # Calculate hours by engineer and classification
        for entry in time_entries:
            if entry.engineer is None or entry.work_order_id is None:
                continue
                
            engineer = entry.engineer
            classification = work_order_classification.get(entry.work_order_id, 'Billable')
            
            engineer_hours[engineer] += entry.hours_worked
            classification_hours[classification] += entry.hours_worked
            engineer_classification_hours[engineer][classification] += entry.hours_worked
        
        # Calculate percentages
        total_hours = sum(engineer_hours.values())
        engineer_percentage = {eng: (hrs/total_hours*100 if total_hours > 0 else 0) 
                              for eng, hrs in engineer_hours.items()}
        classification_percentage = {cls: (hrs/total_hours*100 if total_hours > 0 else 0) 
                                    for cls, hrs in classification_hours.items()}
        
        # Get daily/weekly data for timeline chart
        timeline_data = {}
        if time_frame in ['week', 'month']:
            # Daily aggregation for week or month view
            for i in range((end_date - start_date).days + 1):
                current_date = start_date + timedelta(days=i)
                timeline_data[current_date.strftime('%Y-%m-%d')] = 0
                
            for entry in time_entries:
                if entry.work_date:
                    date_key = entry.work_date.strftime('%Y-%m-%d')
                    if date_key in timeline_data:
                        timeline_data[date_key] += entry.hours_worked
        else:
            # Weekly aggregation for quarter or year view
            current_week_start = start_date - timedelta(days=start_date.weekday())
            while current_week_start <= end_date:
                week_end = current_week_start + timedelta(days=6)
                week_key = f"{current_week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}"
                timeline_data[week_key] = 0
                current_week_start += timedelta(days=7)
                
            for entry in time_entries:
                if entry.work_date:
                    entry_week_start = entry.work_date - timedelta(days=entry.work_date.weekday())
                    entry_week_end = entry_week_start + timedelta(days=6)
                    week_key = f"{entry_week_start.strftime('%b %d')} - {entry_week_end.strftime('%b %d')}"
                    if week_key in timeline_data:
                        timeline_data[week_key] += entry.hours_worked
        
        # Prepare chart data in JSON format
        chart_data = {
            'engineers': list(engineer_hours.keys()),
            'engineerHours': list(engineer_hours.values()),
            'engineerPercentage': list(engineer_percentage.values()),
            'classifications': list(classification_hours.keys()),
            'classificationHours': list(classification_hours.values()),
            'classificationPercentage': list(classification_percentage.values()),
            'timelineLabels': list(timeline_data.keys()),
            'timelineData': list(timeline_data.values()),
            'engineerClassificationData': engineer_classification_hours
        }
        
        return render_template(
            'admin_dashboard.html',
            time_frame=time_frame,
            start_date=start_date,
            end_date=end_date,
            chart_data=chart_data,
            total_hours=total_hours
        )
    
    except Exception as e:
        # Log the exception
        print(f"Error in admin_dashboard: {e}")
        
        # Return an error page or a simplified dashboard with no data
        return render_template(
            'admin_dashboard.html',
            time_frame='month',
            start_date=datetime.now().date() - timedelta(days=30),
            end_date=datetime.now().date(),
            chart_data={
                'engineers': [],
                'engineerHours': [],
                'engineerPercentage': [],
                'classifications': ['Contract/Project', 'Billable', 'Non-Billable'],
                'classificationHours': [0, 0, 0],
                'classificationPercentage': [0, 0, 0],
                'timelineLabels': [],
                'timelineData': [],
                'engineerClassificationData': {}
            },
            total_hours=0,
            error_message=f"An error occurred: {str(e)}"
        )

@app.route('/accounting')
@login_required
@admin_required
def accounting():
    """Optimized accounting page showing work orders with engineer hours and costs"""
    try:
        # Get filter parameters
        status_filter = request.args.get('status', 'all')
        sort_by = request.args.get('sort_by', 'id')
        order = request.args.get('order', 'asc')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))  # Limit to 50 work orders per page
        
        # Build base query with pagination
        query = WorkOrder.query
        if status_filter != 'all':
            query = query.filter_by(status=status_filter)
        
        # Apply sorting at database level
        valid_sort_columns = {
            'id': WorkOrder.id,
            'customer_work_order_number': WorkOrder.customer_work_order_number,
            'rmj_job_number': WorkOrder.rmj_job_number,
            'description': WorkOrder.description,
            'status': WorkOrder.status,
            'owner': WorkOrder.owner,
        }
        
        if sort_by in valid_sort_columns:
            sort_column = valid_sort_columns[sort_by]
            if order == 'desc':
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())
        
        # Paginate the results
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        work_orders = pagination.items
        
        if not work_orders:
            # Return empty results if no work orders
            return render_template('accounting.html', 
                                 work_orders=[],
                                 accounting_data=[],
                                 summary_stats={
                                     'total_work_orders': 0,
                                     'open_count': 0,
                                     'completed_count': 0,
                                     'closed_count': 0,
                                     'grand_total_hours': 0,
                                     'grand_total_cost': 0
                                 },
                                 status_filter=status_filter,
                                 sort_by=sort_by,
                                 order=order,
                                 worker_roles={},
                                 user_rates={},
                                 pagination=pagination)
        
        # Get work order IDs for efficient querying
        work_order_ids = [wo.id for wo in work_orders]
        
        # Single query to get all time entries for these work orders
        time_entries = TimeEntry.query.filter(
            TimeEntry.work_order_id.in_(work_order_ids)
        ).all()
        
        # Group time entries by work order ID for efficient lookup
        time_entries_by_wo = {}
        for entry in time_entries:
            if entry.work_order_id not in time_entries_by_wo:
                time_entries_by_wo[entry.work_order_id] = []
            time_entries_by_wo[entry.work_order_id].append(entry)
        
        # Get worker roles and user rates (cached)
        worker_roles = {role.id: role for role in WorkerRole.query.filter_by(is_active=True).all()}
        users = User.query.all()
        user_rates = {}
        
        for user in users:
            if hasattr(user, 'worker_role_id') and user.worker_role_id and user.worker_role_id in worker_roles:
                user_rates[user.full_name or user.username] = worker_roles[user.worker_role_id].hourly_rate
            else:
                user_rates[user.full_name or user.username] = 0.0
        
        # Process work orders efficiently
        accounting_data = []
        total_summary_hours = 0
        total_summary_cost = 0
        
        for work_order in work_orders:
            # Get time entries for this work order (from our grouped data)
            wo_time_entries = time_entries_by_wo.get(work_order.id, [])
            
            # Calculate engineer breakdown
            engineer_data = process_work_order_entries(wo_time_entries, user_rates)
            
            # Calculate totals
            total_hours = sum(data['hours'] for data in engineer_data.values())
            total_cost = sum(data['cost'] for data in engineer_data.values())
            
            # Add calculated data to work order object
            work_order.engineer_breakdown = engineer_data
            work_order.total_hours = total_hours
            work_order.total_cost = total_cost
            
            # Add to summary
            total_summary_hours += total_hours
            total_summary_cost += total_cost
            
            accounting_data.append({
                'work_order': work_order,
                'engineer_breakdown': engineer_data,
                'total_hours': total_hours,
                'total_cost': total_cost
            })
        
        # Get summary statistics (optimized)
        summary_stats = get_accounting_summary_stats(status_filter)
        summary_stats['current_page_hours'] = total_summary_hours
        summary_stats['current_page_cost'] = total_summary_cost
        
        return render_template('accounting.html', 
                             work_orders=work_orders,
                             accounting_data=accounting_data,
                             summary_stats=summary_stats,
                             status_filter=status_filter,
                             sort_by=sort_by,
                             order=order,
                             worker_roles=worker_roles,
                             user_rates=user_rates,
                             pagination=pagination)
                             
    except Exception as e:
        # Log the error and return a safe fallback
        print(f"Error in accounting route: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal safe page
        return render_template('accounting.html', 
                             work_orders=[],
                             accounting_data=[],
                             summary_stats={
                                 'total_work_orders': 0,
                                 'open_count': 0,
                                 'completed_count': 0,
                                 'closed_count': 0,
                                 'grand_total_hours': 0,
                                 'grand_total_cost': 0
                             },
                             status_filter='all',
                             sort_by='id',
                             order='asc',
                             worker_roles={},
                             user_rates={},
                             pagination=None,
                             error_message="An error occurred loading the accounting data.")

@app.route('/accounting_optimized')
@login_required
@admin_required
def accounting_optimized():
    """Ultra-optimized accounting page using raw SQL for complex calculations"""
    try:
        from sqlalchemy import text
        
        # Get filter parameters
        status_filter = request.args.get('status', 'all')
        sort_by = request.args.get('sort_by', 'id')
        order = request.args.get('order', 'asc')
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 50)), 100)  # Cap at 100
        
        # Build the WHERE clause for status filter
        status_where = ""
        if status_filter != 'all':
            status_where = f"AND wo.status = '{status_filter}'"
        
        # Build ORDER BY clause
        valid_sort_columns = {
            'id': 'wo.id',
            'customer_work_order_number': 'wo.customer_work_order_number',
            'rmj_job_number': 'wo.rmj_job_number',
            'description': 'wo.description',
            'status': 'wo.status',
            'owner': 'wo.owner',
            'total_cost': 'COALESCE(totals.total_cost, 0)'
        }
        
        order_by = valid_sort_columns.get(sort_by, 'wo.id')
        order_direction = 'DESC' if order == 'desc' else 'ASC'
        
        # Calculate offset for pagination
        offset = (page - 1) * per_page
        
        # Raw SQL query with pre-calculated totals (much faster)
        query_sql = text(f"""
        WITH work_order_totals AS (
            SELECT 
                te.work_order_id,
                COUNT(te.id) as entry_count,
                SUM(te.hours_worked) as total_hours,
                SUM(
                    te.hours_worked * 
                    COALESCE(te.rate_at_time_of_entry, 0)
                ) as total_cost
            FROM time_entry te
            GROUP BY te.work_order_id
        )
        SELECT 
            wo.id,
            wo.customer_work_order_number,
            wo.rmj_job_number,
            wo.description,
            wo.status,
            wo.owner,
            wo.project_cost,
            COALESCE(totals.total_hours, 0) as total_hours,
            COALESCE(totals.total_cost, 0) as total_cost,
            COALESCE(totals.entry_count, 0) as entry_count
        FROM work_order wo
        LEFT JOIN work_order_totals totals ON wo.id = totals.work_order_id
        WHERE 1=1 {status_where}
        ORDER BY {order_by} {order_direction}
        LIMIT {per_page} OFFSET {offset}
        """)
        
        # Count query for pagination
        count_sql = text(f"""
        SELECT COUNT(*) as total
        FROM work_order wo
        WHERE 1=1 {status_where}
        """)
        
        # Execute queries
        with db.engine.connect() as conn:
            result = conn.execute(query_sql)
            work_order_data = result.fetchall()
            
            count_result = conn.execute(count_sql)
            total_count = count_result.fetchone()[0]
        
        # Convert to work order objects with calculated data
        work_orders = []
        for row in work_order_data:
            work_order = WorkOrder.query.get(row.id)
            if work_order:
                work_order.total_hours = float(row.total_hours)
                work_order.total_cost = float(row.total_cost)
                work_order.entry_count = row.entry_count
                work_orders.append(work_order)
        
        # Create pagination object manually
        has_prev = page > 1
        has_next = (page * per_page) < total_count
        prev_num = page - 1 if has_prev else None
        next_num = page + 1 if has_next else None
        pages = (total_count + per_page - 1) // per_page
        
        class SimplePagination:
            def __init__(self):
                self.page = page
                self.per_page = per_page
                self.total = total_count
                self.pages = pages
                self.has_prev = has_prev
                self.has_next = has_next
                self.prev_num = prev_num
                self.next_num = next_num
                self.items = work_orders
            
            def iter_pages(self, left_edge=2, left_current=2, right_current=3, right_edge=2):
                last = self.pages
                for num in range(1, last + 1):
                    if num <= left_edge or \
                       (self.page - left_current - 1 < num < self.page + right_current) or \
                       num > last - right_edge:
                        yield num
        
        pagination = SimplePagination()
        
        # Get summary stats efficiently
        summary_stats = {
            'total_work_orders': total_count,
            'open_count': WorkOrder.query.filter_by(status='Open').count(),
            'completed_count': WorkOrder.query.filter_by(status='Complete').count(),
            'closed_count': WorkOrder.query.filter_by(status='Closed').count(),
            'current_page_hours': sum(wo.total_hours for wo in work_orders),
            'current_page_cost': sum(wo.total_cost for wo in work_orders)
        }
        
        # Get user rates (cached)
        user_rates = get_cached_user_rates()
        
        return render_template('accounting.html',
                             work_orders=work_orders,
                             accounting_data=[],  # Not needed with pre-calculated data
                             summary_stats=summary_stats,
                             status_filter=status_filter,
                             sort_by=sort_by,
                             order=order,
                             worker_roles={},
                             user_rates=user_rates,
                             pagination=pagination)
                             
    except Exception as e:
        print(f"Error in optimized accounting route: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to minimal safe page
        return render_template('accounting.html',
                             work_orders=[],
                             accounting_data=[],
                             summary_stats={'total_work_orders': 0, 'open_count': 0, 'completed_count': 0, 'closed_count': 0, 'current_page_hours': 0, 'current_page_cost': 0},
                             status_filter='all',
                             sort_by='id',
                             order='asc',
                             worker_roles={},
                             user_rates={},
                             pagination=None,
                             error_message="An error occurred loading the accounting data.")

@app.route('/accounting_minimal')
@login_required
@admin_required
def accounting_minimal():
    """Minimal accounting page for emergency use"""
    try:
        # Only load basic work order data without complex calculations
        work_orders = WorkOrder.query.limit(20).all()
        
        for wo in work_orders:
            wo.total_hours = 0
            wo.total_cost = 0
            wo.engineer_breakdown = {}
        
        summary_stats = {
            'total_work_orders': WorkOrder.query.count(),
            'open_count': 0,
            'completed_count': 0,
            'closed_count': 0,
            'current_page_hours': 0,
            'current_page_cost': 0
        }
        
        return render_template('accounting.html',
                             work_orders=work_orders,
                             accounting_data=[],
                             summary_stats=summary_stats,
                             status_filter='all',
                             sort_by='id',
                             order='asc',
                             worker_roles={},
                             user_rates={},
                             pagination=None,
                             minimal_mode=True)
                             
    except Exception as e:
        return f"<h1>Accounting System Error</h1><p>Please contact administrator. Error: {str(e)}</p>"

@app.route('/accounting/export')
@login_required
@admin_required
def export_accounting():
    """Export accounting data to Excel"""
    status_filter = request.args.get('status', 'all')
    
    # Get work orders based on filter
    if status_filter == 'all':
        work_orders = WorkOrder.query.all()
    else:
        work_orders = WorkOrder.query.filter_by(status=status_filter).all()
    
    # Get worker roles and user rates (same logic as above)
    worker_roles = {role.id: role for role in WorkerRole.query.filter_by(is_active=True).all()}
    users = User.query.all()
    user_rates = {}
    
    for user in users:
        if hasattr(user, 'worker_role_id') and user.worker_role_id and user.worker_role_id in worker_roles:
            user_rates[user.full_name or user.username] = worker_roles[user.worker_role_id].hourly_rate
        else:
            user_rates[user.full_name or user.username] = 0.0
    
    # Prepare data for Excel export
    export_data = []
    
    for work_order in work_orders:
        time_entries = TimeEntry.query.filter_by(work_order_id=work_order.id).all()
        
        # Group by engineer
        engineer_data = {}
        for entry in time_entries:
            engineer = entry.engineer
            if not engineer:
                continue
                
            if engineer not in engineer_data:
                engineer_data[engineer] = {
                    'hours': 0,
                    'rate': user_rates.get(engineer, 0.0)
                }
            engineer_data[engineer]['hours'] += entry.hours_worked
        
        # Create rows for export
        if engineer_data:
            for engineer, data in engineer_data.items():
                cost = data['hours'] * data['rate']
                export_data.append({
                    'Work Order ID': work_order.id,
                    'Customer WO#': work_order.customer_work_order_number or '',
                    'RMJ Job#': work_order.rmj_job_number,
                    'Description': work_order.description,
                    'Status': work_order.status,
                    'Owner': work_order.owner or '',
                    'Engineer': engineer,
                    'Hours': round(data['hours'], 2),
                    'Hourly Rate': data['rate'],
                    'Total Cost': round(cost, 2)
                })
        else:
            # Work order with no time entries
            export_data.append({
                'Work Order ID': work_order.id,
                'Customer WO#': work_order.customer_work_order_number or '',
                'RMJ Job#': work_order.rmj_job_number,
                'Description': work_order.description,
                'Status': work_order.status,
                'Owner': work_order.owner or '',
                'Engineer': 'No time entries',
                'Hours': 0,
                'Hourly Rate': 0,
                'Total Cost': 0
            })
    
    # Create Excel file
    df = pd.DataFrame(export_data)
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Accounting Data')
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Accounting Data']
        
        # Add formatting
        money_format = workbook.add_format({'num_format': '$#,##0.00'})
        
        # Apply money format to cost columns
        worksheet.set_column('I:I', 12, money_format)  # Hourly Rate
        worksheet.set_column('J:J', 12, money_format)  # Total Cost
        
    output.seek(0)
    
    filename = f"accounting_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(output, 
                     download_name=filename, 
                     as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.route('/accounting/work_order/<int:work_order_id>')
@login_required
@admin_required
def accounting_work_order_detail(work_order_id):
    """Detailed accounting view for a specific work order"""
    work_order = WorkOrder.query.get_or_404(work_order_id)
    
    # Get all time entries for this work order
    time_entries = TimeEntry.query.filter_by(work_order_id=work_order.id).order_by(
        TimeEntry.work_date.desc(), TimeEntry.engineer
    ).all()
    
    # Get worker roles and user rates
    worker_roles = {role.id: role for role in WorkerRole.query.filter_by(is_active=True).all()}
    users = User.query.all()
    user_rates = {}
    
    for user in users:
        if hasattr(user, 'worker_role_id') and user.worker_role_id and user.worker_role_id in worker_roles:
            user_rates[user.full_name or user.username] = worker_roles[user.worker_role_id].hourly_rate
        else:
            user_rates[user.full_name or user.username] = 0.0
    
    # Calculate detailed breakdown
    engineer_summary = {}
    daily_breakdown = {}
    
    for entry in time_entries:
        engineer = entry.engineer
        if not engineer:
            continue
        
        # USE HISTORICAL RATE IF AVAILABLE
        if entry.rate_at_time_of_entry is not None:
            rate = entry.rate_at_time_of_entry
            role = entry.role_at_time_of_entry or 'Unknown'
        else:
            rate = user_rates.get(engineer, 0.0)
            role = 'Current Rate'
        
        cost = entry.hours_worked * rate
        
        # Engineer summary
        if engineer not in engineer_summary:
            engineer_summary[engineer] = {
                'total_hours': 0,
                'total_cost': 0,
                'entries': []
            }
        
        engineer_summary[engineer]['total_hours'] += entry.hours_worked
        engineer_summary[engineer]['total_cost'] += cost
        engineer_summary[engineer]['entries'].append({
            'date': entry.work_date,
            'hours': entry.hours_worked,
            'cost': cost,
            'rate': rate,
            'role': role,  # ADD THIS
            'description': entry.description,
            'time_in': entry.time_in,
            'time_out': entry.time_out
        })
        
        # Daily breakdown
        date_str = entry.work_date.strftime('%Y-%m-%d')
        if date_str not in daily_breakdown:
            daily_breakdown[date_str] = {
                'date': entry.work_date,
                'total_hours': 0,
                'total_cost': 0,
                'engineers': {}
            }
        
        daily_breakdown[date_str]['total_hours'] += entry.hours_worked
        daily_breakdown[date_str]['total_cost'] += cost
        
        if engineer not in daily_breakdown[date_str]['engineers']:
            daily_breakdown[date_str]['engineers'][engineer] = {
                'hours': 0,
                'cost': 0
            }
        
        daily_breakdown[date_str]['engineers'][engineer]['hours'] += entry.hours_worked
        daily_breakdown[date_str]['engineers'][engineer]['cost'] += cost
    
    # Calculate totals
    total_hours = sum(summary['total_hours'] for summary in engineer_summary.values())
    total_cost = sum(summary['total_cost'] for summary in engineer_summary.values())
    
    return render_template('accounting_work_order_detail.html',
                         work_order=work_order,
                         engineer_summary=engineer_summary,
                         daily_breakdown=sorted(daily_breakdown.values(), 
                                              key=lambda x: x['date'], reverse=True),
                         total_hours=total_hours,
                         total_cost=total_cost,
                         user_rates=user_rates)

@app.route('/accounting/update_cost/<int:work_order_id>', methods=['POST'])
@login_required
@admin_required
def update_work_order_cost(work_order_id):
    """Update project cost for a work order"""
    try:
        work_order = WorkOrder.query.get_or_404(work_order_id)
        new_cost = float(request.form.get('project_cost', 0))
        work_order.project_cost = new_cost
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/classify_work_orders', methods=['GET', 'POST'])
@login_required
@admin_required
def classify_work_orders():
    """Bulk classify work orders"""
    message = None
    if request.method == 'POST':
        # Get all work orders
        work_orders = WorkOrder.query.all()
        
        # Initialize counters
        billable_count = 0
        non_billable_count = 0
        skipped_count = 0
        
        # Process each work order
        for work_order in work_orders:
            customer_wo = work_order.customer_work_order_number
            rmj_job = work_order.rmj_job_number
            
            # Check if it's already classified
            if hasattr(work_order, 'classification') and work_order.classification:
                if work_order.classification == 'Contract/Project':
                    # Skip Contract/Project as requested
                    skipped_count += 1
                    continue
                    
            # Apply classification rules
            if customer_wo in ['0', '00', '000', '0000']:
                work_order.classification = 'Non-Billable'
                non_billable_count += 1
            elif customer_wo and rmj_job and len(customer_wo.strip()) == 6:
                work_order.classification = 'Billable'
                billable_count += 1
            else:
                # Default to Billable for anything else
                work_order.classification = 'Billable'
                billable_count += 1
        
        # Commit changes
        db.session.commit()
        
        # Log the operation
        log_change(
            session.get('user_id'),
            "Bulk Classification",
            "WorkOrder",
            None,
            f"Bulk classified work orders: {billable_count} as Billable, {non_billable_count} as Non-Billable, {skipped_count} skipped"
        )
        db.session.commit()
        
        message = f"Classification complete! {billable_count} work orders set as Billable, {non_billable_count} set as Non-Billable, {skipped_count} skipped (Contract/Project)."
        
    return render_template('admin_classify_work_orders.html', message=message)


@app.route('/admin/dashboard/engineer_entries/<engineer>')
@login_required
@admin_required
def get_engineer_entries(engineer):
    """Get time entries for a specific engineer"""
    try:
        # Get time frame from request parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        # Parse dates or use defaults
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)
        
        if not start_date or not end_date:
            return jsonify({'error': 'Invalid date parameters'}), 400
            
        # Query for this engineer's time entries in the date range
        entries = (TimeEntry.query
                  .filter(TimeEntry.engineer == engineer)
                  .filter(TimeEntry.work_date >= start_date)
                  .filter(TimeEntry.work_date <= end_date)
                  .order_by(TimeEntry.work_date.desc())
                  .all())
                  
        # Format the entries for JSON response
        result = []
        for entry in entries:
            # Get work order details
            work_order = WorkOrder.query.get(entry.work_order_id)
            classification = getattr(work_order, 'classification', 'Billable') if work_order else 'Billable'
            
            result.append({
                'id': entry.id,
                'engineer': entry.engineer if entry.engineer else engineer,
                'work_date': entry.work_date.strftime('%Y-%m-%d'),
                'time_in': entry.time_in.strftime('%H:%M'),
                'time_out': entry.time_out.strftime('%H:%M'),
                'hours_worked': round(entry.hours_worked, 2),
                'description': entry.description,
                'work_order': {
                    'id': work_order.id if work_order else None,
                    'rmj_job_number': work_order.rmj_job_number if work_order else 'N/A',
                    'customer_work_order_number': work_order.customer_work_order_number if work_order else 'N/A',
                    'description': work_order.description if work_order else 'N/A',
                    'classification': classification
                }
            })
            
        return jsonify({
            'engineer': engineer,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'entries': result,
            'total_hours': sum(entry['hours_worked'] for entry in result)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/dashboard/classification_entries/<classification>')
@login_required
@admin_required
def get_classification_entries(classification):
    """Get time entries for a specific classification"""
    try:
        # Get time frame from request parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        # Parse dates or use defaults
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)
        
        if not start_date or not end_date:
            return jsonify({'error': 'Invalid date parameters'}), 400
        
        # Subquery to get work orders with this classification
        if hasattr(WorkOrder, 'classification'):
            # If classification field exists
            work_orders = WorkOrder.query.filter_by(classification=classification).all()
        else:
            # Temporary classification logic
            if classification == 'Contract/Project':
                work_orders = WorkOrder.query.filter(
                    WorkOrder.customer_work_order_number.like('______'),  # 6 digits
                    WorkOrder.rmj_job_number.isnot(None)
                ).all()
            elif classification == 'Non-Billable':
                work_orders = WorkOrder.query.filter(
                    WorkOrder.customer_work_order_number.in_(['0', '00', '000', '0000'])
                ).all()
            else:  # Billable - default case
                work_orders = WorkOrder.query.filter(
                    ~WorkOrder.customer_work_order_number.in_(['0', '00', '000', '0000']),
                    ~WorkOrder.customer_work_order_number.like('______')
                ).all()
        
        work_order_ids = [wo.id for wo in work_orders]
        
        # Get all time entries for these work orders in the date range
        entries = []
        if work_order_ids:
            entries = (TimeEntry.query
                      .filter(TimeEntry.work_order_id.in_(work_order_ids))
                      .filter(TimeEntry.work_date >= start_date)
                      .filter(TimeEntry.work_date <= end_date)
                      .order_by(TimeEntry.work_date.desc())
                      .all())
        
        # Format the entries for JSON response
        result = []
        for entry in entries:
            # Get work order details
            work_order = WorkOrder.query.get(entry.work_order_id)
            
            result.append({
                'id': entry.id,
                'engineer': entry.engineer if entry.engineer else "Not Specified",  # Add fallback
                'work_date': entry.work_date.strftime('%Y-%m-%d'),
                'time_in': entry.time_in.strftime('%H:%M'),
                'time_out': entry.time_out.strftime('%H:%M'),
                'hours_worked': round(entry.hours_worked, 2),
                'description': entry.description,
                'work_order': {
                    'id': work_order.id if work_order else None,
                    'rmj_job_number': work_order.rmj_job_number if work_order else 'N/A',
                    'customer_work_order_number': work_order.customer_work_order_number if work_order else 'N/A',
                    'description': work_order.description if work_order else 'N/A',
                    'classification': classification  # Ensure classification is included
                }
            })
            
        return jsonify({
            'classification': classification,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'entries': result,
            'total_hours': sum(entry['hours_worked'] for entry in result)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/dashboard/all_classification_entries')
@login_required
@admin_required
def get_all_classification_entries():
    """Get time entries for all classifications - client will filter by classification"""
    try:
        # Get time frame from request parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        # Parse dates or use defaults
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)
        
        if not start_date or not end_date:
            return jsonify({'error': 'Invalid date parameters'}), 400
        
        # Get all time entries for the period
        entries = (TimeEntry.query
                  .filter(TimeEntry.work_date >= start_date)
                  .filter(TimeEntry.work_date <= end_date)
                  .order_by(TimeEntry.work_date.desc())
                  .all())
                  
        # Format all entries for JSON response
        result = []
        for entry in entries:
            # Get work order details
            work_order = WorkOrder.query.get(entry.work_order_id)
            classification = getattr(work_order, 'classification', 'Billable') if work_order else 'Billable'
            
            result.append({
                'id': entry.id,
                'engineer': entry.engineer if entry.engineer else "Not Specified",
                'work_date': entry.work_date.strftime('%Y-%m-%d'),
                'time_in': entry.time_in.strftime('%H:%M'),
                'time_out': entry.time_out.strftime('%H:%M'),
                'hours_worked': round(entry.hours_worked, 2),
                'description': entry.description,
                'work_order': {
                    'id': work_order.id if work_order else None,
                    'rmj_job_number': work_order.rmj_job_number if work_order else 'N/A',
                    'customer_work_order_number': work_order.customer_work_order_number if work_order else 'N/A',
                    'description': work_order.description if work_order else 'N/A',
                    'classification': classification
                }
            })
            
        return jsonify({
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'entries': result,
            'total_hours': sum(entry['hours_worked'] for entry in result)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/bulk_delete_time_entries', methods=['GET', 'POST'])
@login_required
@admin_required
def bulk_delete_time_entries():
    """Bulk delete time entries"""
    message = None
    if request.method == 'POST':
        selected_ids = request.form.getlist('time_entry_ids')
        if not selected_ids:
            message = "No time entries selected."
        else:
            try:
                # Convert selected IDs to integers.
                ids = list(map(int, selected_ids))
            except Exception:
                message = "Error processing selected IDs."
            else:
                # Query for the time entries with the selected IDs.
                entries = TimeEntry.query.filter(TimeEntry.id.in_(ids)).all()
                count = len(entries)
                for entry in entries:
                    db.session.delete(entry)
                db.session.commit()
                message = f"Deleted {count} time entry record(s)."
    # On GET (or after deletion) show all time entries.
    time_entries = TimeEntry.query.order_by(TimeEntry.id).all()
    return render_template('admin_bulk_delete_time_entries.html', time_entries=time_entries, message=message)


@app.route('/admin/import_time_entries', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_import_time_entries():
    """Import time entries from Excel"""
    message = None
    if request.method == 'POST':
        if 'file' not in request.files:
            message = "No file part provided."
            return render_template('admin_import_time_entries.html', message=message)
            
        file = request.files['file']
        if file.filename == '':
            message = "No file selected."
            return render_template('admin_import_time_entries.html', message=message)
        
        try:
            # Load the entire workbook so we can iterate over its sheets.
            xls = pd.ExcelFile(file)
            count = 0  # Counter for the number of imported time entries

            # Process each sheet in the workbook.
            for sheet_name in xls.sheet_names:
                try:
                    # Read only the first 50 rows after the header row
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=1, nrows=50)
                except ValueError:
                    continue

                if df.empty:
                    continue

                # Ensure required columns are present.
                required_columns = ['Engineer', 'Date:', 'WO#', 'Job Number', 'Hours']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    continue

                # Process each row in the current sheet.
                for index, row in df.iterrows():
                    # Retrieve and process the Job Number from the row.
                    job_number_raw = row.get("Job Number")
                    if pd.isna(job_number_raw):
                        continue

                    try:
                        if isinstance(job_number_raw, (int, float)):
                            job_number = str(int(job_number_raw))
                        else:
                            job_number = str(job_number_raw).strip()
                    except Exception:
                        continue

                    # Look up the work order using the Job Number.
                    work_order = WorkOrder.query.filter_by(rmj_job_number=job_number).first()
                    if not work_order:
                        continue

                    # Process the Engineer field.
                    engineer = row.get("Engineer")
                    if pd.isna(engineer):
                        continue

                    # Process the Date field.
                    date_value = row.get("Date:")
                    if pd.isna(date_value):
                        continue

                    try:
                        if isinstance(date_value, datetime):
                            work_date = date_value.date()
                        else:
                            work_date = pd.to_datetime(date_value).date()
                    except Exception:
                        continue

                    # Process the hours field.
                    hours = row.get("Hours")
                    if pd.isna(hours):
                        continue

                    try:
                        hours_float = float(hours)
                    except Exception:
                        continue

                    if hours_float <= 0:
                        continue  # Skip rows with zero or negative hours

                    # Set the time_in value to midnight.
                    time_in_value = time(0, 0)
                    # Calculate time_out by adding the hours worked.
                    time_out_dt = datetime.combine(work_date, time_in_value) + timedelta(hours=hours_float)
                    time_out_value = time_out_dt.time()
                    current_role, current_rate = get_user_role_and_rate(engineer)

                    # Create and add the new TimeEntry record.
                    new_entry = TimeEntry(
                        work_order_id=work_order.id,
                        engineer=engineer,
                        work_date=work_date,
                        time_in=time_in_value,
                        time_out=time_out_value,
                        hours_worked=hours_float,
                        description="Imported time entry",
                        role_at_time_of_entry=current_role,    # ADD THIS
                        rate_at_time_of_entry=current_rate     # ADD THIS
                    )
                    db.session.add(new_entry)
                    count += 1

            db.session.commit()
            message = f"Imported {count} time entry record(s)."
        except Exception as e:
            message = f"Error processing file: {str(e)}"
    
    return render_template('admin_import_time_entries.html', message=message)


# =============================================================================
# TIME ENTRY REASSIGNMENT ROUTES
# =============================================================================
@app.route('/workorder/reassign_entries', methods=['GET', 'POST'])
@login_required
@admin_required
def reassign_entries():
    """Reassign time entries from one work order to another"""
    if request.method == 'POST':
        source_id = request.form.get('source_work_order_id')
        target_id = request.form.get('target_work_order_id')
        if not source_id or not target_id:
            return "Please select both source and target work orders", 400
        entries = TimeEntry.query.filter_by(work_order_id=source_id).all()
        for entry in entries:
            entry.work_order_id = int(target_id)
        db.session.commit()
        return redirect(url_for('index'))
    else:
        work_orders = WorkOrder.query.all()
        return render_template('reassign_entries.html', work_orders=work_orders)


@app.route('/workorder/reassign_entries_selected', methods=['POST'])
@login_required
@admin_required
def reassign_entries_selected():
    """Selectively reassign specific time entries"""
    if request.method == 'POST':
        try:
            # Get target work order ID and entry IDs
            target_id = request.form.get('target_id')
            entries_json = request.form.get('entries_json')
            
            if not target_id or not entries_json:
                flash("Missing required parameters", "danger")
                return redirect(url_for('reassign_entries'))
            
            # Parse the entry IDs
            entry_ids = json.loads(entries_json)
            
            if not entry_ids:
                flash("No entries selected", "warning")
                return redirect(url_for('reassign_entries'))
            
            # Get all the time entries
            entries = TimeEntry.query.filter(TimeEntry.id.in_(entry_ids)).all()
            
            # Check if any entries are locked due to JL/JT checkboxes
            locked_entries = [entry for entry in entries if entry.entered_on_jl or entry.entered_on_jt]
            if locked_entries:
                locked_count = len(locked_entries)
                flash(f"Cannot reassign {locked_count} time entries: They have been entered into the accounting system (JL/JT checked).", "danger")
                return redirect(url_for('reassign_entries'))
            
            # Get the target work order
            target_work_order = WorkOrder.query.get_or_404(int(target_id))
            
            # Track source work order IDs for logging
            source_work_order_ids = set()
            
            # Update each entry
            for entry in entries:
                source_work_order_ids.add(entry.work_order_id)
                entry.work_order_id = int(target_id)
            
            # Commit the changes
            db.session.commit()
            
            # Log the reassignment
            source_ids_str = ", ".join(map(str, source_work_order_ids))
            log_change(
                session.get('user_id'),
                "Reassigned Time Entries",
                "TimeEntry",
                None,
                f"Reassigned {len(entries)} time entries from work order(s) {source_ids_str} to work order {target_id}"
            )
            db.session.commit()
            
            # Flash a success message
            flash(f"Successfully reassigned {len(entries)} time entries to work order {target_work_order.rmj_job_number}", "success")
            
            # Redirect to the target work order's detail page
            return redirect(url_for('work_order_detail', work_order_id=target_id))
            
        except Exception as e:
            # Log the error and show an error message
            print(f"Error in reassign_entries_selected: {e}")
            flash(f"Error reassigning time entries: {str(e)}", "danger")
            return redirect(url_for('index'))

    # If not POST, redirect to the main reassign entries page
    return redirect(url_for('reassign_entries'))


# =============================================================================
# API ROUTES
# =============================================================================
@app.route('/api/work_order/<int:work_order_id>/time_entries', methods=['GET'])
@login_required
def get_work_order_time_entries(work_order_id):
    """API endpoint to get time entries for a specific work order"""
    try:
        # Get the work order
        work_order = WorkOrder.query.get_or_404(work_order_id)
        
        # Get all time entries for this work order
        entries = TimeEntry.query.filter_by(work_order_id=work_order_id).all()
        
        # Format the entries for JSON response
        result = []
        for entry in entries:
            result.append({
                'id': entry.id,
                'work_order_id': entry.work_order_id,
                'engineer': entry.engineer,
                'work_date': entry.work_date.strftime('%Y-%m-%d'),
                'time_in': entry.time_in.strftime('%H:%M'),
                'time_out': entry.time_out.strftime('%H:%M'),
                'hours_worked': float(entry.hours_worked),
                'description': entry.description
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# CONTEXT PROCESSORS
# =============================================================================
@app.context_processor
def inject_user_model():
    """Inject User model into all templates"""
    return dict(User=User)

@app.context_processor
def inject_datetime():
    from datetime import datetime, timedelta
    return {
        'datetime': datetime,
        'timedelta': timedelta,
        'today_date': datetime.now().date()
    }


# =============================================================================
# APP INITIALIZATION
# =============================================================================
# Initialize the database when the app starts
with app.app_context():
    populate_user_full_names()

# Add this to the APP INITIALIZATION section

def initialize_notification_settings():
    """Initialize default notification settings if they don't exist"""
    notification_types = [
        'report_upload', 'report_approval', 'status_change', 
        'hours_threshold', 'scheduled_date', 'new_work_order'
    ]
    
    for notification_type in notification_types:
        # Check if setting already exists
        setting = NotificationSetting.query.filter_by(notification_type=notification_type).first()
        if not setting:
            # Create default setting
            enabled = notification_type == 'report_upload'  # Only enable report upload by default
            default_options = {}
            
            # Set default options based on notification type
            if notification_type == 'report_upload':
                default_options = {
                    'report_keywords': ['report', 'assessment', 'evaluation']
                }
            elif notification_type == 'hours_threshold':
                default_options = {
                    'warning_threshold': 80,
                    'exceeded_alert': True,
                    'include_work_order_owner': True
                }
            elif notification_type == 'scheduled_date':
                default_options = {
                    'days_before': 3,
                    'include_owner': True
                }
            elif notification_type == 'status_change':
                default_options = {
                    'open_to_complete': True,
                    'complete_to_closed': True,
                    'any_to_open': False
                }
            elif notification_type == 'report_approval':
                default_options = {
                    'send_reminder': False,
                    'reminder_days': 3
                }
            elif notification_type == 'new_work_order':
                default_options = {
                    'high_priority': True,
                    'medium_priority': True,
                    'low_priority': False
                }
            
            new_setting = NotificationSetting(
                notification_type=notification_type,
                enabled=enabled,
                options=default_options
            )
            db.session.add(new_setting)
            
            # Add default recipient for report_upload if applicable
            if notification_type == 'report_upload' and 'REPORT_NOTIFICATION_EMAIL' in app.config:
                email = app.config['REPORT_NOTIFICATION_EMAIL']
                if email:
                    recipient = NotificationRecipient(
                        notification_setting=new_setting,
                        email=email
                    )
                    db.session.add(recipient)
    
    db.session.commit()
    print("Notification settings initialized")

# Add this to the app initialization
# Add this to the app initialization
with app.app_context():
    try:
        db.create_all()
        
        # Add the worker_role_id column if it doesn't exist
        column_added = add_worker_role_id_column()
        
        # ADD THIS LINE - Add the new TimeEntry columns
        add_time_entry_role_columns()
        add_project_cost_column()
        add_database_indexes()
        
        if column_added:
            print("Database column ready - you can now uncomment the model fields")
        
        populate_user_full_names()
        initialize_notification_settings()
        
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error during app initialization: {e}")

def start_scheduler():
    """Start the background task scheduler"""
    import threading
    import time
    
    def run_scheduler():
        """Run scheduled tasks"""
        while True:
            try:
                with app.app_context():
                    # Check for scheduled date reminders every day
                    check_scheduled_date_reminders()
                    
                    # Check for report approval reminders
                    # This would check for reports that were uploaded but not approved
                    # and send reminders if necessary
                    
            except Exception as e:
                print(f"Error in scheduler: {e}")
            
            # Sleep for 1 hour before checking again
            time.sleep(3600)
    
    # Start the scheduler in a background thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    print("Scheduler started")

# Start the scheduler when the app starts
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    start_scheduler()


# Run the application if executed directly
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
        
# config_additions.py - ADD THESE SETTINGS TO YOUR FLASK APP CONFIGURATION

import os



# File upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads', 'tickets')
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Make sure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

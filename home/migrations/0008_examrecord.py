# Generated by Django 4.2.3 on 2023-07-06 03:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("home", "0007_delete_examrecord"),
    ]

    operations = [
        migrations.CreateModel(
            name="ExamRecord",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("warningcount", models.PositiveIntegerField()),
                ("image", models.ImageField(upload_to="exam_images/")),
                ("timeofjoining", models.DateTimeField()),
                ("is_session_running", models.BooleanField(default=False)),
                (
                    "score",
                    models.DecimalField(
                        blank=True, decimal_places=2, max_digits=5, null=True
                    ),
                ),
                (
                    "college",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="home.college"
                    ),
                ),
                (
                    "student",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="home.student"
                    ),
                ),
                (
                    "subject",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="home.collegesubjects",
                    ),
                ),
            ],
        ),
    ]

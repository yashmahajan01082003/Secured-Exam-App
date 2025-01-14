# Generated by Django 4.2.3 on 2023-07-05 07:26

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="College",
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
                ("college_id", models.CharField(max_length=10, unique=True)),
                ("name", models.CharField(max_length=100)),
                ("city", models.CharField(max_length=100)),
            ],
            options={
                "verbose_name_plural": "Colleges",
            },
        ),
    ]

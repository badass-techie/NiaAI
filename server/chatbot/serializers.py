from rest_framework import serializers
from .models import Group, User


class UserSerializer(serializers.ModelSerializer):
    group_id = serializers.PrimaryKeyRelatedField(queryset=Group.objects.all(), source='group')
    group_name = serializers.CharField(source='group.name', read_only=True)

    class Meta:
        model = User
        fields = ['address', 'purpose_statement', 'group_id', 'group_name']


class GroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = Group
        fields = ['id', 'name']


class GroupDetailSerializer(serializers.ModelSerializer):
    user_addresses = serializers.SerializerMethodField()

    class Meta:
        model = Group
        fields = ['id', 'name', 'user_addresses']

    def get_user_addresses(self, obj):
        return [user.address for user in obj.user_set.all()]

